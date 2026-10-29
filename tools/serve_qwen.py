"""Single-GPU Qwen2-VL server for the original RRB planner protocols.

Run behind an SSH tunnel on loopback. Supports non-streaming text, inline images,
JSON-schema constrained decoding and completion-only stop strings. Unsupported
options fail explicitly. Importing this module does not load a model or CUDA.
"""
import base64
from contextlib import asynccontextmanager
import io
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
import uuid
import warnings

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from responsible_robot_bench.planner.limits import context_limit, fits_context
from responsible_robot_bench.planner.qwen_protocol import (CompletionStop, completion_result, only_keys,
                               response_schema, stop_strings, trim_stop)

MODEL_ID = 'Qwen/Qwen2-VL-7B-Instruct'
MODEL_PATH = os.environ.get('RRB_MODEL_PATH', MODEL_ID)
MAX_IMAGE_URL_CHARS = 20_000_000
MAX_IMAGE_PIXELS = 20_000_000
lock = threading.Lock()
runtime = None


def load_runtime():
    import torch
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn,
    )
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', local_files_only=True,
    ).eval()
    native_limit = model.config.max_position_embeddings
    limit = context_limit(os.environ.get('RRB_MAX_CONTEXT_TOKENS'), native_limit)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, max_pixels=1024 * 28 * 28,
    )
    # Cache vocabulary preprocessing once; every request gets its own parser.
    tokenizer_data = build_token_enforcer_tokenizer_data(
        processor.tokenizer, vocab_size=min(len(processor.tokenizer), model.config.vocab_size),
    )
    return SimpleNamespace(
        torch=torch, model=model, processor=processor,
        process_vision_info=process_vision_info, tokenizer_data=tokenizer_data,
        build_prefix=build_transformers_prefix_allowed_tokens_fn,
        context_limit=limit, native_limit=native_limit,
    )


@asynccontextmanager
async def lifespan(app):
    global runtime
    runtime = load_runtime()
    yield
    runtime = None


app = FastAPI(title='RRB local Qwen2-VL inference', lifespan=lifespan)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    model: str = MODEL_ID
    messages: list[dict] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0, ge=0, le=2)
    stream: bool = False
    response_format: dict | None = None
    stop: str | list[str] | None = None

    @field_validator('response_format')
    @classmethod
    def validate_response_format(cls, value):
        response_schema(value)
        return value

    @field_validator('stop')
    @classmethod
    def validate_stop(cls, value):
        stop_strings(value)
        return value


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    # Omit input values (which can contain the entire prompt or image payload).
    detail = '; '.join(f'{".".join(map(str, item["loc"]))}: {item["msg"]}'
                       for item in error.errors())
    return JSONResponse(status_code=400, content={'error': {
        'message': detail, 'type': 'invalid_request_error', 'param': None, 'code': None}})


@app.exception_handler(HTTPException)
async def http_error(request: Request, error: HTTPException):
    return JSONResponse(status_code=error.status_code, content={'error': {
        'message': str(error.detail),
        'type': 'invalid_request_error' if error.status_code < 500 else 'server_error',
        'param': None, 'code': None}})


def prepare_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('Expected at least one message')
    result = []
    for message in messages:
        only_keys(message, {'role', 'content'}, 'message')
        role, content = message.get('role'), message.get('content')
        if role not in ('system', 'user', 'assistant'):
            raise ValueError('Unsupported message role')
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        if not isinstance(content, list):
            raise ValueError('Content must be text or a list of parts')
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError('Content parts must be objects')
            if part.get('type') == 'text':
                only_keys(part, {'type', 'text'}, 'text part')
                value = part.get('text')
                if not isinstance(value, str):
                    raise ValueError('Text part must contain a string')
                parts.append({'type': 'text', 'text': value})
            elif part.get('type') == 'image_url':
                only_keys(part, {'type', 'image_url'}, 'image part')
                image_url = part.get('image_url')
                only_keys(image_url, {'url', 'detail'}, 'image_url')
                if image_url.get('detail', 'auto') != 'auto':
                    raise ValueError('Only image_url.detail=auto is supported')
                url = image_url.get('url')
                if not isinstance(url, str) or len(url) > MAX_IMAGE_URL_CHARS:
                    raise ValueError('Image URL must be a string within the per-image byte limit')
                header, separator, encoded = url.partition(',')
                formats = {'data:image/png;base64': 'PNG', 'data:image/jpeg;base64': 'JPEG'}
                if not separator or header not in formats:
                    raise ValueError('Only inline PNG/JPEG images are accepted')
                with warnings.catch_warnings():
                    warnings.simplefilter('error', Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as im:
                        if im.format != formats[header]:
                            raise ValueError('Image format does not match its data URL MIME type')
                        if im.width * im.height > MAX_IMAGE_PIXELS:
                            raise ValueError('Image dimensions exceed the per-image pixel limit')
                        parts.append({'type': 'image', 'image': im.convert('RGB')})
            else:
                raise ValueError('Unsupported content part')
        result.append({'role': role, 'content': parts})
    return result


def schema_parser(schema, alphabet):
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser

    # 0 disables LMFE's implicit 20-item array cap. Context/output tokens, rather
    # than formatter heuristics, bound unconstrained array and whitespace length.
    config = CharacterLevelParserConfig(alphabet=alphabet, max_json_array_length=0,
                                       max_consecutive_whitespaces=sys.maxsize,
                                       force_json_field_order=False)
    return JsonSchemaParser(schema, config=config)


def generation_options(request, prompt_tokens, rt):
    kwargs = {'max_new_tokens': request.max_tokens, 'do_sample': request.temperature > 0,
              'num_beams': 1, 'num_return_sequences': 1}
    if request.temperature > 0:
        kwargs['temperature'] = request.temperature
    stops = stop_strings(request.stop)
    if stops:
        kwargs['stopping_criteria'] = [CompletionStop(rt.processor.tokenizer, prompt_tokens, stops)]
    schema = response_schema(request.response_format)
    if schema is not None:
        parser = schema_parser(schema, rt.tokenizer_data.tokenizer_alphabet)
        config = parser.config
        kwargs['prefix_allowed_tokens_fn'] = rt.build_prefix(rt.tokenizer_data, parser)
        # LMFE 0.11.3's TokenEnforcer replaces parser.config during construction.
        # Restore the explicit per-request settings before generation starts.
        parser.config = config
    return kwargs


def validate_completed_json(answer, schema, alphabet):
    """Detect backend failures without repairing or fabricating model output."""
    json.loads(answer)
    parser = schema_parser(schema, alphabet)
    for character in answer:
        if character not in parser.get_allowed_characters():
            raise ValueError('Generated output violates the requested schema')
        parser = parser.add_character(character)
    if not parser.can_end():
        raise ValueError('Generated output is incomplete')


def ready_runtime():
    if runtime is None:
        raise HTTPException(503, 'Model is not loaded')
    return runtime


@app.get('/health')
def health():
    rt = ready_runtime()
    cuda = rt.torch.cuda
    return {'status': 'ready', 'model': MODEL_ID, 'dtype': str(rt.model.dtype),
            'gpu': cuda.get_device_name(0), 'allocated_bytes': cuda.memory_allocated(0),
            'reserved_bytes': cuda.memory_reserved(0),
            'peak_allocated_bytes': cuda.max_memory_allocated(0),
            'free_bytes': cuda.mem_get_info(0)[0],
            'context_limit_tokens': rt.context_limit,
            'model_context_limit_tokens': rt.native_limit, 'busy': lock.locked()}


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model'}]}


@app.post('/v1/chat/completions')
def chat(request: ChatRequest):
    if request.stream:
        raise HTTPException(400, 'Streaming is not implemented')
    if request.model not in {MODEL_ID, 'Qwen2-VL-7B-Instruct'}:
        raise HTTPException(400, 'This server hosts Qwen2-VL-7B-Instruct only')
    try:
        messages = prepare_messages(request.messages)
    except (ValueError, TypeError, KeyError, OSError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(400, str(error)) from error
    rt = ready_runtime()
    with lock, rt.torch.inference_mode():
        text = rt.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = rt.process_vision_info(messages)
        inputs = rt.processor(text=[text], images=images, videos=videos,
                              padding=True, truncation=False, return_tensors='pt')
        prompt_tokens = inputs.input_ids.shape[1]
        if not fits_context(prompt_tokens, request.max_tokens, rt.context_limit):
            raise HTTPException(400, f'Context requires {prompt_tokens} prompt + '
                                f'{request.max_tokens} completion tokens; limit is {rt.context_limit}')
        kwargs = generation_options(request, prompt_tokens, rt)
        inputs = inputs.to('cuda:0')
        generated = rt.model.generate(**inputs, **kwargs)
        tokens = generated[:, prompt_tokens:]
        answer = rt.processor.batch_decode(tokens, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        token_ids = tokens[0].tolist()
        completion_tokens = len(token_ids)
        stops = stop_strings(request.stop)
        stopped = trim_stop(answer, stops)[1]
        answer, finish_reason = completion_result(
            answer, stops, token_ids, request.max_tokens, rt.model.generation_config.eos_token_id,
        )
        schema = response_schema(request.response_format)
        # A user stop or exhausted token budget can legitimately leave partial JSON.
        # For a natural EOS, do not report malformed output as a completed schema.
        if schema is not None and finish_reason == 'stop' and not stopped:
            try:
                validate_completed_json(answer, schema, rt.tokenizer_data.tokenizer_alphabet)
            except (ValueError, TypeError, KeyError) as error:
                raise HTTPException(500, 'Constrained decoding produced invalid JSON') from error
    return {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
            'created': int(time.time()), 'model': MODEL_ID,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer},
                         'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                      'total_tokens': prompt_tokens + completion_tokens}}

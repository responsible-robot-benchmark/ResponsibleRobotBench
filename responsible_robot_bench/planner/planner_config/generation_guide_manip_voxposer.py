

user1 = kwargs.pop('prompt')
new_query = '# Query:' + user1.split('# Query:')[-1]
user1 = ''.join(user1.split('# Query:')[:-1]).strip()
user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
assistant1 = f'Got it. I will complete what you give me next.'
user2 = new_query
# handle given context (this was written originally for completion endpoint)
if user1.split('\n')[-4].startswith('objects = ['):
    obj_context = user1.split('\n')[-4]
    # remove obj_context from user1
    user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
    # add obj_context to user2
    user2 = obj_context.strip() + '\n' + user2
messages=[
    {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
    {"role": "user", "content": user1},
    {"role": "assistant", "content": assistant1},
    {"role": "user", "content": user2},
]
# Third-party acknowledgements and licenses

ResponsibleRobotBench builds on the following research software. We thank their
authors and contributors, and retain the applicable notices in the source tree.

The root MIT grant covers only the original RRB files listed in
[LICENSE_SCOPE.md](LICENSE_SCOPE.md). It does not replace any component terms
or establish permission to redistribute restricted upstream components.

| Project | Use in ResponsibleRobotBench | License / notice |
| --- | --- | --- |
| [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench) | Manipulation environment and embodied-planner infrastructure | Attribution and applicable upstream terms retained |
| [RLBench](https://github.com/stepjam/RLBench) | Task environment, demonstrations, and manipulation infrastructure | [Retained component agreement](licenses/RLBench-LICENSE) |
| [PyRep](https://github.com/stepjam/PyRep) | CoppeliaSim Python binding | [MIT](licenses/PyRep-LICENSE) |
| [PointFlowMatch](https://github.com/robot-learning-freiburg/PointFlowMatch) | Imitation-learning integration | [GPL v3](policies/PointFlowMatch/LICENSE) |
| [CLIP](https://github.com/openai/CLIP) | Pretrained visual/language encoders and tokenizer | [MIT](licenses/CLIP-LICENSE) |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | Policy data and training utilities, through the PFM dependency | [MIT](licenses/DiffusionPolicy-LICENSE), also retained in the installed dependency |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | Optional YOLO11 detection backend, installed separately | [Upstream license](https://github.com/ultralytics/ultralytics/blob/v8.3.221/LICENSE); model weights are downloaded separately |
| [CoppeliaSim](https://www.coppeliarobotics.com/) | Simulation engine, installed separately | Publisher's terms; engine binaries are not bundled |

PFM installation pins its Diffusion Policy dependency and includes the packaging
patch in `patches/diffusion-policy-packaging.patch`. Original license texts and
source-file attributions remain intact. Component licenses apply to their
respective code and assets; this notice does not replace those terms.

The task scenes and robot assets retain their applicable upstream attribution.
Project figures and the Franka/UR5 demonstrations in `docs/assets/` accompany
ResponsibleRobotBench. Manufacturer and model names identify the experimental
platforms and evaluated systems.

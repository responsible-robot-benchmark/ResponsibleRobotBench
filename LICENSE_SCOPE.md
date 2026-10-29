# License scope

ResponsibleRobotBench contains original software and components from several
research projects. The root [MIT License](LICENSE) is scoped to the original
files below; it is not a license for the entire combined simulator and policy
distribution. Existing third-party notices take precedence for their components.

## Original RRB software: MIT

The MIT grant covers the original contents of:

- `rrb/`: task registry, planner client, configuration handling, execution
  adapters, episode records, metrics, and validation tools.
- `tools/`, `tests/`, `configs/`, and `constraints/`: original command-line
  tools, regression tests, experiment configurations, and dependency constraints.
- `.github/workflows/`, `.env.example`, `.gitignore`, `.gitattributes`,
  `pyproject.toml`, and root `requirements-*.txt` files.
- `README.md`, `LICENSE_SCOPE.md`, `THIRD_PARTY_NOTICES.md`, and the original
  Markdown guides in `docs/`. Images and videos in `docs/assets/` are excluded.

Imports or adapters do not grant new rights to the software they invoke. The
Python wheel installs only the `rrb` package and includes this scope statement.
It does not bundle the simulator environment, PFM, or task scene assets.

## Components retaining their own terms

- `policies/PointFlowMatch/` retains its [GPL-3.0 license](policies/PointFlowMatch/LICENSE),
  including applicable terms for modifications and distribution. The root MIT
  grant does not cover PFM-derived code. Obligations for a combined program
  depend on how the components are integrated, not only on directory layout.
- `thirdparty/PyRep/` retains the [PyRep MIT license](licenses/PyRep-LICENSE).
- `responsible_robot_bench/` and retained root research utilities contain EmbodiedBench,
  RLBench, and other derived implementations. They are excluded from the root
  MIT grant. In particular, the RLBench-derived `amsolver` environment retains
  the [component agreement](licenses/RLBench-LICENSE); specifically identified
  BSD elements retain their BSD terms.
- Embedded CLIP components retain the [CLIP MIT license](licenses/CLIP-LICENSE).
- Diffusion Policy, installed separately by the policy workflow, retains its
  MIT license. The small packaging patch in `patches/` does not replace that
  upstream license.
- CoppeliaSim and model weights are installed separately under their publishers'
  terms. Their installation is not a license grant from this project.

Official upstream links and component locations are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Original notices remain in the
source tree; modifications do not erase upstream rights.

## Scenes, data, figures, and videos

Simulator `.ttm`/`.ttt` scenes, robot models, textures, datasets, checkpoints,
and project media are not covered by the software MIT grant. They retain any
applicable source terms. No blanket asset or dataset license is granted here.

## Distribution status

The original RRB software has an explicit MIT grant. The project owner reports
having checked the redistribution permissions for the included third-party
code and assets. The applicable upstream terms and retained component notices
continue to govern those components; this statement does not replace them or
apply the root MIT grant to third-party material.

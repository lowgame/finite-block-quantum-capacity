# Third-party material

The unchanged integer verifier in `capacity_experiment/baseline/zenodo_v1_0/artifact/verifier/verify.py` comes from A. Krohn-Grimberghe's *A certified lower bound on the quantum-capacity threshold of the depolarizing channel*, version 1.0, [Zenodo DOI 10.5281/zenodo.21962925](https://doi.org/10.5281/zenodo.21962925). That deposit declares [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The published verifier is separate from the new interval implementations.

The finite-channel kernel is independently implemented from Appendix E of arXiv:2608.15870v2. Gaussian-envelope seeds are heuristic choices inspired by arXiv:2609.39747v2. Bosonic coefficient lists in `capacity_experiment/candidates/families.py` are attributed to equations (10)-(11) and Appendix B of arXiv:2607.27449v1, provided under CC BY 4.0; their finite-qubit lifts are labeled as experimental candidates.

Dependencies are installed separately and retain their own licenses. Third-party licenses apply to the corresponding material.

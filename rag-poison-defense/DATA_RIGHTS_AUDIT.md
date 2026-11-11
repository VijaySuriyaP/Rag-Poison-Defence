# Dataset rights and redistribution audit

**Checked:** 2026-11-02. This is a repository-level research note, not legal advice.

| Source | Evidence checked | Finding | Project handling |
|---|---|---|---|
| Tensor Trust data | [Upstream data repository](https://github.com/HumanCompatibleAI/tensor-trust-data), [Hugging Face mirror card](https://huggingface.co/datasets/qxcv/tensor-trust), [open upstream license issue #3](https://github.com/HumanCompatibleAI/tensor-trust-data/issues/3) (opened 2024-11-06; no maintainer response shown when checked), and the [project paper/data page](https://tensortrust.ai/paper) | No `LICENSE` file was present in the upstream repository's visible root file listing, and the mirror card contains no license metadata. The project paper page says the data are released, but the material checked did not identify an explicit dataset license. An upstream issue asks the maintainers to add one. A stated intent to release permissively is not treated here as a grant of redistribution rights. | Keep downloaded samples local/private; do not publish records, derived text excerpts, or a data bundle until the maintainers provide written clarification or an explicit license. Access-code fields and exact text occurrences are redacted by the preparation pipeline. |
| Wikimedia Wikipedia | `wikimedia/wikipedia`, `20231101.en`; dataset manifest and Wikimedia licensing notes | Source page identifies CC-BY-SA-3.0/GFDL. Attribution/share-alike obligations may apply to copied or adapted text. | Keep source IDs/URLs and license metadata; review attribution and share-alike requirements before redistribution. |
| deepset prompt-injections | `deepset/prompt-injections`; dataset card and manifest | Manifest records Apache-2.0. Verify source-card terms/version before external redistribution. | Preserve source attribution and manifest metadata. |
| BIPIA public mirror | `geodesic-research/bipia`; dataset card and manifest | Manifest records CC-BY-SA-4.0 for the public mirror. Confirm that the mirror's terms apply to the precise records used. | Preserve attribution and source metadata; review share-alike obligations. |

## Follow-up

Before publishing generated artifacts outside this workspace, verify whether report tables, videos, model artifacts, or examples include memorized/excerpted source text. Obtain written clarification from Tensor Trust maintainers or exclude Tensor Trust from any distributable artifact. This audit does not authorize redistribution and should be refreshed before release.

# Ethics and responsible-use policy

## Scope

This is local defensive research on indirect prompt injection in retrieval-augmented generation (RAG). It evaluates ingestion scanners on public benchmark records and unmodified Wikimedia article text. Benchmark attack text is treated only as inert data; it is never executed or sent to an external target.

## Dataset safeguards

- Follow the upstream dataset licenses, attribution requirements, access terms, and takedown requests. Check terms again before redistribution.
- The Tensor Trust mirror has no license metadata in its card. Do not redistribute its downloaded records unless upstream rights are confirmed.
- Tensor Trust access-code fields are omitted and occurrences in prompt text are redacted. Do not attempt to recover them or identify dataset contributors.
- Treat public emails, names, and other personal-looking text as sensitive. Do not attempt re-identification or publish unnecessary row-level excerpts.
- Downloaded dataset artifacts are local-only by default; reports publish aggregate metrics and counts, not raw prompt contents.

## Responsible use and limitations

- Do not use this project to probe systems without explicit authorization and a scoped test plan.
- Scanner scores are triage signals, not ground truth. Evaluate false positives, review burden, and utility before deployment.
- The current victim output is a deterministic simulator diagnostic, not ASR. It is not a real-model safety result or production security certification.
- The cost model is based on explicit assumptions, not observed expenditure.
- A public benchmark cannot represent every RAG corpus, threat actor, language, or deployment configuration.

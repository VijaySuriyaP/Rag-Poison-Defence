# Threat model

## System and trust boundaries

```text
Untrusted uploader / URL
        |
        v
[parse + normalize] -- [content scanner] -- [provenance verifier]
        |                    |                      |
        +--------------------+----------------------+
                             v
                    [quarantine / review]
                             |
                    allowed documents only
                             v
       [chunker] -> [embedder] -> [vector index] -> [top-k retrieval]
                                                        |
                                       untrusted context delimiter
                                                        v
                                          [victim model / simulator]
                                                        |
                                                 [output guard]
                                                        v
                                                   answer + logs
```

## Attacker capabilities

The attacker can submit text documents through an assumed public ingestion interface and can choose wording, formatting, metadata claims, language, and placement of instructions. The attacker cannot modify source code, signing keys, scanner configuration, the vector index directly, or the evaluation harness. Public BIPIA and Tensor Trust records represent the attacker-controlled examples in this study.

## Assets and goals

Assets include answer integrity, clean-document utility, index integrity, source provenance, secrets excluded from context, and availability of the RAG service. Public benchmark examples cover indirect instructions, answer manipulation, prompt extraction, and other benchmark-defined attack goals. Tensor Trust access-code fields are excluded and occurrences in saved prompt text are redacted.

## Trust tiers

- **T0:** internal document with a valid signed manifest and matching content hash.
- **T1:** content from an allowlisted domain with verifiable source metadata.
- **T2:** unknown source or insufficient provenance.
- **T3:** anonymous or user-uploaded document.

Trust changes thresholds and retrieval eligibility; it does not imply content is safe.

## Assumptions and limitations

The repository models ingestion and retrieval locally. A deterministic simulated-victim diagnostic is reported separately and is not ASR or an estimate of any production model's behavior. Retrieval selection uses development data only; held-out paired-context Recall@k is reported separately. Real-world parsing, OCR, PDF extraction, multilingual coverage, source verification, and human review are only partially represented.

## Out of scope

Production deployment, real data exfiltration, external tool execution, credential theft, real-user targeting, executing embedded code, bypassing deployed security controls, and claims that benchmark results directly predict a commercial model's vulnerability.

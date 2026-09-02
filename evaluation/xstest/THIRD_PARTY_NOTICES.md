# Third-Party Notices

## XSTest

- Project: https://github.com/paul-rottger/xstest
- Revision: `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`
- Paper: https://aclanthology.org/2024.naacl-long.301/
- Dataset: `data/xstest_prompts.csv`
- License: Creative Commons Attribution 4.0 International
- License text: `data/LICENSE.XSTest`

The prompt dataset is redistributed unmodified. Its SHA256 is recorded in code and
`data/README.md`. The three-class evaluation prompt in `xstest_common.py` is adapted
from the official `evaluation/classify_completions_gpt.py` script, with the remote
OpenAI call replaced by local Hugging Face generation and an explicit one-label
output instruction.

XSTest is by Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio,
Federico Bianchi, and Dirk Hovy. See the paper for the requested citation.

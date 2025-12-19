"""Entry point for running code-transformer as a module.

Enables: python -m code_transformer
"""

from code_transformer.cli.commands import app

if __name__ == "__main__":
    app()

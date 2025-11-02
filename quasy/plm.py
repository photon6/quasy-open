def plm_generate(full_import: str) -> str:
    """
    Generate a PLM (Pretrained Language Model) import statement.

    Args:
        full_import (str): The full import path of the PLM.

    Returns:
        str: A formatted PLM import statement.
    """
    return f"from {full_import} import *"
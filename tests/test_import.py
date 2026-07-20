def test_import_puremacro():
    import re

    import puremacro
    assert hasattr(puremacro, "__version__")
    assert isinstance(puremacro.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", puremacro.__version__), puremacro.__version__

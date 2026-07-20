> 🇬🇧 [English](../CREDENTIALS.md) · 🇪🇸 Español

# Credenciales

> Disponible a partir de puremacro **0.66.0**.

`puremacro.credentials` es el punto único desde el cual todos los recuperadores
de datos con clave API en puremacro leen su clave. Las claves se resuelven en
orden de prioridad:

1. **Argumento explícito** — `credentials.get("fred", explicit=...)` o el
   parámetro `api_key=` de cualquier recuperador tiene prioridad sobre todo lo
   demás.
2. **Variables de entorno** — cada servicio tiene un alias primario
   (`FRED_API_KEY`) y uno secundario con prefijo `PUREMACRO_`
   (`PUREMACRO_FRED_API_KEY`); gana el primero que se encuentre.
3. **Archivo de configuración TOML** — `~/.puremacro/credentials.toml`
   (reemplazable mediante `$PUREMACRO_CREDENTIALS_FILE` o `$XDG_CONFIG_HOME`).
4. **None** — `get()` devuelve `None`; `require()` lanza
   `MissingCredentialError` con un mensaje orientado a la acción del
   investigador.

## Inicio rápido

```python
import puremacro.credentials as creds

# Ver qué está configurado (nunca expone los valores reales de las claves):
creds.status()
#       service  configured           source                                         description                                 signup_url
# 0        fred        True  env:FRED_API_KEY            FRED + ALFRED real-time macro data (...)  https://fred.stlouisfed.org/...
# 1         bea       False          missing               BEA NIPA / regional / industry tables  https://apps.bea.gov/API/signup/
# ...

# Resolver una clave (None si no se encuentra):
key = creds.get("anthropic")

# O requerirla (lanza una excepción con un mensaje informativo):
key = creds.require("anthropic")
# MissingCredentialError: LLM-scored narrative kernels (...) needs an
# API key. Checked env vars (in order): ANTHROPIC_API_KEY,
# PUREMACRO_ANTHROPIC_API_KEY. Checked config file:
# /Users/you/.puremacro/credentials.toml (not found). Get a free key
# at: https://console.anthropic.com/settings/keys
```

## Formato del archivo de configuración

`~/.puremacro/credentials.toml` (opcional; créelo si prefiere no definir
variables de entorno):

```toml
[fred]
api_key = "abc123..."

[bea]
api_key = "..."

[anthropic]
api_key = "sk-ant-..."

[openai]
api_key = "sk-..."

[census]
api_key = "..."
```

Las secciones ausentes recurren a las variables de entorno. El archivo se lee
una sola vez por proceso y queda en caché. Un TOML malformado emite un
`UserWarning` y cae al modo de solo variables de entorno — nunca bloquea la
resolución de credenciales.

## Servicios reconocidos

| Servicio    | Utilizado por                                  | Registro                                                 |
|-------------|------------------------------------------------|----------------------------------------------------------|
| `fred`      | `fetch.fred`, `fetch.fred_states`, FRB Phil    | https://fred.stlouisfed.org/docs/api/api_key.html        |
| `bea`       | `fetch.bea_cainc`, `fetch.bea_industry_shares` | https://apps.bea.gov/API/signup/                         |
| `anthropic` | `narrative.scoring.llm` (proveedor Anthropic)  | https://console.anthropic.com/settings/keys              |
| `openai`    | `narrative.scoring.llm` (proveedor OpenAI)     | https://platform.openai.com/api-keys                     |
| `census`    | (declarado anticipadamente; sin consumidor actual) | https://api.census.gov/data/key_signup.html          |

> ⚠️ El servicio `census` está declarado de forma anticipada (registrado en
> `SERVICES` para futuros conectores directos a la API del Census, p. ej., ACS
> o series BFS por estado no reflejadas en FRED). El `fetch.census_bfs` actual
> obtiene datos del BFS del Census a través de FRED — véase la fila `fred`
> anterior.

## Para implementadores (añadir un nuevo recuperador)

```python
from puremacro import credentials

def fetch_my_thing(*, api_key: str | None = None) -> pd.DataFrame:
    key = credentials.require("my_service", explicit=api_key)
    # ... use key in your HTTP calls ...
```

La prueba de análisis sintáctico AST
`tests/test_credentials/test_no_direct_env_get_in_fetch.py`
hace fallar la compilación si se lee `os.environ.get("*_API_KEY")` directamente
en `puremacro/{fetch,narrative/scoring,narrative/indices,instruments}/`.

Para añadir un nuevo servicio reconocido, agregue una entrada
`ServiceCredentialSpec` en `puremacro/credentials.py::SERVICES`. La prueba del
registro de servicios verifica que cada entrada disponga de los campos
requeridos y de una URL de registro HTTPS.

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

Un servicio se autentica con **usuario y contraseña** en lugar de con un
token único (el Banco Central de Chile). En ese caso, `get()` / `require()`
resuelven el *usuario* y `get_password()` resuelve la *contraseña*; véase
[Credenciales de dos partes](#credenciales-de-dos-partes-bcch).

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

[banxico]
api_key = "..."

[inegi]
api_key = "..."

[bcch]
user = "..."
password = "..."
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
| `banxico`   | `fetch.realtime.banxico`                       | https://www.banxico.org.mx/SieAPIRest/service/v1/token_req.html |
| `inegi`     | `fetch.realtime.inegi`                         | https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/ |
| `bcch`      | `fetch.realtime.bcch`                          | https://si3.bcentral.cl/estadisticas/principal1/registro/index.html |

> ⚠️ El servicio `census` está declarado de forma anticipada (registrado en
> `SERVICES` para futuros conectores directos a la API del Census, p. ej., ACS
> o series BFS por estado no reflejadas en FRED). El `fetch.census_bfs` actual
> obtiene datos del BFS del Census a través de FRED — véase la fila `fred`
> anterior.

### Conectores latinoamericanos de tiempo real

Los tres servicios incorporados en la versión 3.4.0 sustentan los conectores
de instantánea descritos en
[`docs/es/real_time_data.md`](real_time_data.md). El Banco Central do Brasil
(`bcb`) no requiere credencial alguna y, por tanto, no tiene entrada.

| Servicio | Variables de entorno (en orden de resolución) | Claves TOML | Por dónde viaja la credencial |
|---|---|---|---|
| `banxico` | `BANXICO_API_KEY`, `BMX_TOKEN`, `PUREMACRO_BANXICO_API_KEY` | `[banxico].api_key` | la **cabecera** de la solicitud `Bmx-Token` |
| `inegi` | `INEGI_API_KEY`, `PUREMACRO_INEGI_API_KEY` | `[inegi].api_key` | un segmento de la **ruta de la URL** |
| `bcch` | usuario: `BCCH_API_USER`, `PUREMACRO_BCCH_API_USER` · contraseña: `BCCH_API_PASS`, `PUREMACRO_BCCH_API_PASS` | `[bcch].user`, `[bcch].password` (`[bcch].api_key` se sigue leyendo como usuario, por compatibilidad) | ambas en la **cadena de consulta** (`user=`, `pass=`) |

Dado que el token de INEGI viaja en la ruta y el par del BCCh en la cadena de
consulta, cualquiera de ellos puede acabar en una URL que arrastre una
excepción o una línea de registro. El conector del BCCh depura el usuario y la
contraseña — en texto plano y codificados para URL — de sus advertencias y de
la URL que urllib adjunta a sus errores; trate como secreta cualquier URL que
usted mismo imprima.

### Credenciales de dos partes (`bcch`)

```python
import puremacro.credentials as creds

creds.get("bcch")           # el USUARIO    (BCCH_API_USER / [bcch].user)
creds.get_password("bcch")  # la CONTRASEÑA (BCCH_API_PASS / [bcch].password)
creds.require("bcch")       # lanza una excepción salvo que AMBAS se resuelvan
```

`PASSWORD_ENV_VARS` enumera los servicios cuya credencial es un par. Una
contraseña por sí sola nunca satisface una búsqueda: si solo está definida
`BCCH_API_PASS`, `get("bcch")` devuelve `None` y `status()` reporta `bcch`
como no configurado. Si solo está definido el usuario, `status()` señala la
mitad ausente — `env:BCCH_API_USER (no password: set BCCH_API_PASS or
PUREMACRO_BCCH_API_PASS or [bcch].password)` — y `require("bcch")` lanza una
excepción nombrando ambas listas.

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

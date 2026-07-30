# Guía de despliegue — paso a paso

Esta guía está escrita para seguirla desde cero. Al terminar tendrás la API
corriendo en internet con una dirección propia.

---

## Parte 0 · Entender qué vas a hacer

Tu modelo hoy vive en un archivo. Para usarlo hay que abrir Python y ejecutarlo.

**Desplegar** significa ponerlo en un servidor de internet que esté siempre
encendido, con una dirección pública, para que cualquier programa pueda
preguntarle sin saber nada de Python.

El servicio que usaremos es **Vercel**. Es gratis para proyectos como este y se
conecta directo a GitHub: cada vez que subas un cambio, se actualiza solo.

---

## Parte 1 · Probarlo en tu computadora (10 minutos)

Antes de subir nada a internet, conviene verlo funcionar en local. Si falla
aquí, fallará allá.

### 1.1 Abre una terminal en la carpeta del proyecto

```powershell
cd C:\Users\FABRIZIO\fastapi-boilerplate
```

### 1.2 Crea un entorno aislado

Un *entorno virtual* es una carpeta donde se instalan las librerías de este
proyecto, sin mezclarse con las del resto de tu sistema.

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Sabrás que funcionó porque tu terminal mostrará `(.venv)` al inicio de la línea.

### 1.3 Instala lo necesario

```powershell
pip install -r requirements.txt
```

Son solo dos librerías de ejecución más las de prueba. Tarda segundos, no
minutos: aquí no hay scikit-learn ni NumPy.

### 1.4 Comprueba que todo está bien

```powershell
pytest -q
```

Deberías ver **`24 passed`**. Si sale eso, el modelo responde exactamente lo que
respondía scikit-learn.

### 1.5 Enciende el servidor

```powershell
uvicorn main:app --reload
```

Verás algo como:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Tu API ya está viva**, solo que únicamente en tu máquina.

### 1.6 Pruébala

Abre en el navegador: **http://localhost:8000/docs**

Esa página se genera sola a partir del código. Para probar:

1. Haz clic en **`POST /predict`**
2. Botón **`Try it out`**
3. El formulario ya viene con un cliente de ejemplo
4. Botón **`Execute`**
5. Baja hasta *Response body*

Deberías ver:

```json
{ "churn_probability": 0.9583, "churn_prediction": 1, "risk_band": "High" }
```

**Ese cliente tiene 95,8 % de probabilidad de irse.** Prueba a cambiar
`contract_type` a `"Two year"` y ejecutar otra vez: la probabilidad se desploma.
Ahí estás viendo el efecto del coeficiente más fuerte del modelo.

Para apagar el servidor: **Ctrl + C** en la terminal.

---

## Parte 2 · Subirlo a internet (15 minutos)

### 2.1 Sube tus cambios a GitHub

Si hiciste modificaciones:

```powershell
git add -A
git commit -m "ajustes"
git push
```

### 2.2 Crea la cuenta de Vercel

1. Entra a **https://vercel.com/signup**
2. Elige **Continue with GitHub**
3. Autoriza el acceso

No necesitas tarjeta de crédito.

### 2.3 Importa el repositorio

1. En el panel de Vercel: **Add New…** → **Project**
2. Busca este repositorio en la lista y pulsa **Import**
   - Si no aparece, pulsa **Adjust GitHub App Permissions** y dale acceso
3. Vercel detectará solo que es un proyecto Python con FastAPI
4. **No cambies nada** de la configuración
5. Pulsa **Deploy**

### 2.4 Espera

Tarda entre uno y dos minutos. Al terminar verás tu dirección.

**Este proyecto ya está desplegado en:**

### https://churn-prediction-api-ruddy.vercel.app

### 2.5 Quita la protección

Vercel activa una capa de autenticación en los proyectos nuevos. Sin quitarla,
la API responde `401 Protected deployment` a cualquiera que no tenga tu sesión
iniciada — un reclutador vería una pantalla de login en lugar de tu API.

Desde el panel: **Settings** → **Deployment Protection** → **Vercel
Authentication** → **Disabled** → **Save**.

O desde la terminal:

```powershell
vercel project protection disable --sso
```

> Piénsalo antes: al desactivarla, cualquiera puede llamar a tu API sin límite
> de peticiones. Para un proyecto de portafolio es lo que quieres, pero conviene
> saberlo. Se vuelve a activar con `enable` en lugar de `disable`.

### 2.6 Compruébalo

Abre https://churn-prediction-api-ruddy.vercel.app/docs y repite la prueba del
paso 1.6. Si devuelve `0.9583`, está funcionando igual que en tu máquina.

---

## Parte 3 · Usarla de verdad

### Desde la terminal

```powershell
curl -X POST https://churn-prediction-api-ruddy.vercel.app/predict `
  -H "Content-Type: application/json" `
  -d '{\"tenure_months\":2,\"monthly_charges\":95.0,\"total_charges\":190.0,\"num_support_tickets\":4,\"age\":30,\"contract_type\":\"Month-to-month\",\"internet_service\":\"Fiber optic\",\"payment_method\":\"Electronic check\",\"gender\":\"Female\",\"has_streaming\":\"No\",\"paperless_billing\":\"Yes\"}'
```

> En PowerShell el acento grave `` ` `` continúa la línea, y las comillas dobles
> internas van escapadas con `\`.

### Desde Python

```python
import requests

cliente = {
    "tenure_months": 2, "monthly_charges": 95.0, "total_charges": 190.0,
    "num_support_tickets": 4, "age": 30,
    "contract_type": "Month-to-month", "internet_service": "Fiber optic",
    "payment_method": "Electronic check", "gender": "Female",
    "has_streaming": "No", "paperless_billing": "Yes",
}

r = requests.post("https://churn-prediction-api-ruddy.vercel.app/predict", json=cliente)
print(r.json())
# {'churn_probability': 0.9583, 'churn_prediction': 1, 'risk_band': 'High'}
```

### Varios clientes a la vez

`POST /predict/batch` acepta una lista de hasta 1 000 clientes y devuelve una
lista de resultados en el mismo orden. Es lo que usarías para puntuar una
cartera completa de una sola llamada.

---

## Si algo sale mal

| Síntoma | Causa probable | Solución |
|---|---|---|
| `'uvicorn' no se reconoce` | El entorno no está activado | `.venv\Scripts\activate` |
| `ModuleNotFoundError: fastapi` | Faltan las dependencias | `pip install -r requirements.txt` |
| `Address already in use` | Ya hay algo en el puerto 8000 | `uvicorn main:app --port 8001` |
| Error 422 al pedir predicción | Un campo falta o tiene un valor no permitido | Revisa la tabla de campos del README |
| El build de Vercel falla | Versión de Python no soportada | Edita `.python-version` y prueba con `3.12` |
| Vercel bloquea el deploy | Exige commits firmados | Ver la nota de abajo |

### Sobre los commits firmados

En tu proyecto `ghl-dashboard` Vercel bloqueó un deploy por exigir commits
firmados. Si vuelve a pasar aquí, tienes dos salidas:

1. **Configurar la firma de commits** en tu Git (más trabajo, pero útil a largo
   plazo).
2. **Usar otro servicio.** [Render](https://render.com) y
   [Hugging Face Spaces](https://huggingface.co/spaces) despliegan APIs de
   Python gratis y sin esa restricción. El código no cambia.

---

## Qué acabas de conseguir

Un modelo entrenado en un notebook, corriendo en internet, respondiendo
consultas de cualquier programa que sepa hacer una petición HTTP.

Eso, en la industria, se llama **poner un modelo en producción**, y es la parte
que la mayoría de proyectos de portafolio no llega a mostrar.

### Ideas para seguir

- Conectar la API al **dashboard** del repo de churn, para que el simulador
  llame al servicio en vez de calcular en el navegador.
- Añadir un endpoint que reciba un **CSV** y devuelva la cartera puntuada.
- Registrar cada consulta para medir cuántas predicciones se piden y con qué
  perfiles — el primer paso hacia monitorizar un modelo en producción.

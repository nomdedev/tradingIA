# 🚀 Cómo Ejecutar Trading IA desde el Escritorio

## ⚠️ IMPORTANTE: Framework GUI Utilizado

**Este proyecto utiliza PySide6 (Qt for Python), NO PyQt6.**
- ✅ Framework correcto: PySide6
- ❌ Framework incorrecto: PyQt6

Si encuentras errores de importación, asegúrate de tener PySide6 instalado:
```bash
pip install PySide6
```

## Opciones para Ejecutar la Aplicación

### 1. Script PowerShell (Recomendado para Windows)
```powershell
# Desde PowerShell, ejecuta:
.\start_platform.ps1
```

**Características:**
- ✅ Verificación automática de dependencias
- ✅ Activación automática del entorno virtual
- ✅ Mensajes de error descriptivos
- ✅ Verificación de archivo .env

### 2. Archivo Batch (.bat)
```cmd
# Desde CMD o haciendo doble clic:
start_platform.bat
```

**Características:**
- ✅ Simple de usar (doble clic)
- ✅ Compatible con CMD
- ✅ Verificación básica de dependencias

### 3. Ejecución Directa (Avanzado)
```bash
# Activar entorno virtual (opcional pero recomendado)
.venv\Scripts\activate

# Ejecutar la aplicación
python src\main_platform.py
```

## 📋 Requisitos Previos

### 1. Python 3.8+
Asegúrate de tener Python instalado:
```cmd
python --version
```

### 2. Dependencias Instaladas
```cmd
pip install -r requirements_platform.txt
```

### 3. Archivo de Configuración (.env)
Crea un archivo `.env` en la raíz del proyecto:
```env
# Copia el contenido de .env.example y configura:
ALPACA_API_KEY=tu_api_key_aqui
ALPACA_SECRET_KEY=tu_secret_key_aqui
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## 🎯 ¿Qué Hace Cada Opción?

### Plataforma Principal (`src/main_platform.py`)
- **Interfaz Gráfica Completa** con PyQt6
- **9 Pestañas** de funcionalidad completa:
  - 📊 Dashboard general
  - 📥 Gestión de datos
  - ⚙️ Configuración de estrategias
  - ▶️ Backtesting
  - 📈 Análisis de resultados
  - 🆚 A/B Testing
  - 📊 Monitoreo en vivo
  - 🔬 Análisis avanzado
  - 📥 Descarga de datos
- **Carga automática** de datos BTC/USD al iniciar
- **Paper Trading** integrado

### Dashboard Web (`dashboard/app.py`)
- **Interfaz web** con Flask
- **Visualizaciones** básicas
- **Análisis simple** de estrategias

## 🔧 Solución de Problemas

### Error: "No se encuentra src\main_platform.py"
- Asegúrate de ejecutar desde la **raíz del proyecto** TradingIA
- La estructura debe ser: `D:\martin\Proyectos\tradingIA\`

### Error: "Dependencias no instaladas"
```cmd
pip install -r requirements_platform.txt
```

### Error: "No se encontró entorno virtual"
- Crea un entorno virtual: `python -m venv .venv`
- Actívalo: `.venv\Scripts\activate`

### La aplicación no se abre
- Verifica los logs en `logs/trading.log`
- Asegúrate de que no hay otra instancia ejecutándose
- En Windows, puede requerir instalación de PyQt6 correctamente

## 📁 Estructura del Proyecto
```
tradingIA/
├── src/main_platform.py      # 🖥️  Aplicación principal (GUI)
├── dashboard/app.py          # 🌐 Dashboard web
├── start_platform.ps1        # 🚀 Script PowerShell
├── start_platform.bat        # 🚀 Script Batch
├── requirements_platform.txt # 📦 Dependencias
└── .env                      # 🔑 Configuración
```

## 🎮 Uso Rápido

1. **Descarga el proyecto** a `D:\martin\Proyectos\tradingIA\`
2. **Instala dependencias**: `pip install -r requirements_platform.txt`
3. **Configura .env** con tus credenciales de Alpaca
4. **Ejecuta**: `.\start_platform.ps1` o doble clic en `start_platform.bat`

¡La plataforma se abrirá automáticamente! 🚀
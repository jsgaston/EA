import ccxt
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

# Descargar recursos necesarios para NLTK
nltk.download('vader_lexicon')

# Función para obtener noticias y realizar análisis de sentimiento
def get_news_sentiment(symbol, api_key, date):
    try:
        newsapi = NewsApiClient(api_key=api_key)
        
        # Obtener noticias relacionadas con el símbolo para la fecha específica
        end_date = date + timedelta(days=1)
        articles = newsapi.get_everything(q=symbol,
                                          from_param=date.strftime('%Y-%m-%d'),
                                          to=end_date.strftime('%Y-%m-%d'),
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=10)
        
        sia = SentimentIntensityAnalyzer()
        
        sentiments = []
        for article in articles['articles']:
            text = article.get('title', '')
            if article.get('description'):
                text += ' ' + article['description']
            
            if text:
                sentiment = sia.polarity_scores(text)
                sentiments.append(sentiment['compound'])
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return avg_sentiment
    except Exception as e:
        print(f"Error al obtener el sentimiento para {symbol} en la fecha {date}: {e}")
        return 0  # Devolver un sentimiento neutral en caso de error

# Crear una instancia del exchange Binance
binance = ccxt.binance()

# Definir el símbolo del mercado y el intervalo de tiempo
symbol = 'ETH/USDT'
timeframe = '1d'
limit = 1000

# Descargar los datos históricos
ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)

# Convertir los datos a un DataFrame de pandas
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Guardar los datos en un archivo CSV
df.to_csv('binance_data.csv', index=False)
print("Datos descargados y guardados en 'binance_data.csv'")

# Cargar los datos descargados
data = pd.read_csv('binance_data.csv')

# Asegurarse de que la columna 'timestamp' esté en formato datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Crear un objeto MinMaxScaler con función rolling
scaler = MinMaxScaler()

# Normalizar los datos de cierre utilizando una función rolling
data['close_normalized'] = data['close'].rolling(window=120, min_periods=1).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)), raw=True)

# Eliminar las filas donde no se tiene suficiente historia para calcular el scaler
data = data.dropna()

# Guardar los datos normalizados en un archivo CSV (opcional)
data.to_csv('binance_data_normalized.csv', index=False)
print("Datos normalizados guardados en 'binance_data_normalized.csv'")

# Cargar el modelo ONNX
model = onnx.load('model_ethusdt.onnx')
onnx.checker.check_model(model)

# Crear una sesión de runtime
ort_session = ort.InferenceSession('model_ethusdt.onnx')

# Preparar los datos para el modelo como ventanas deslizantes
input_name = ort_session.get_inputs()[0].name
sequence_length = 120  # Ajustar esto según el modelo

# Crear una lista para almacenar las predicciones
predictions_list = []

# Definir la fecha inicial para las predicciones
start_date = pd.Timestamp('2024-05-25')
end_date = pd.Timestamp.today()

# NewsAPI Key (regístrate en https://newsapi.org/ para obtener una clave gratuita)
news_api_key = '3b21cf66ca1e48e2b2360fed96529ff9'

# Realizar la inferencia día a día
current_date = start_date
sentiments = []
while current_date < end_date:
    # Seleccionar los últimos 120 días de datos antes de la fecha actual
    end_idx = data[data['timestamp'] < current_date].index[-1]
    start_idx = end_idx - sequence_length + 1
    
    if start_idx < 0:
        print(f"No hay suficientes datos para la fecha {current_date}")
        break
    
    # Extraer la ventana de datos normalizados y desnormalizar
    window_normalized = data['close_normalized'].values[start_idx:end_idx+1]
    window_actual = data['close'].values[start_idx:end_idx+1]
    
    # Calcular min y max dentro de la ventana
    min_close_window = np.min(window_actual)
    max_close_window = np.max(window_actual)
    
    # Preparar los datos para el modelo
    input_window = np.array(window_normalized).astype(np.float32)
    input_window = np.expand_dims(input_window, axis=0)  # Añadir dimensión de batch size
    input_window = np.expand_dims(input_window, axis=2)  # Añadir dimensión de características
    
    # Realizar la inferencia
    output = ort_session.run(None, {input_name: input_window})
    prediction = output[0][0][0]
    
    # Desnormalizar la predicción utilizando min y max de la ventana actual
    prediction = prediction * (max_close_window - min_close_window) + min_close_window
    
    # Obtener el análisis de sentimiento basado en noticias
    sentiment = get_news_sentiment(symbol.split('/')[0], news_api_key, current_date)
    
    # Almacenar la predicción y el sentimiento
    predictions_list.append({
        'date': current_date,
        'prediction': prediction,
        'sentiment': sentiment
    })
    
    sentiments.append({'date': current_date, 'sentiment': sentiment})
    
    # Incrementar la fecha
    current_date += pd.Timedelta(days=1)

# Guardar los sentimientos en un archivo CSV
sentiments_df = pd.DataFrame(sentiments)
sentiments_df.to_csv('daily_sentiments.csv', index=False)
print("Sentimientos diarios guardados en 'daily_sentiments.csv'")

# Convertir la lista de predicciones a un DataFrame
predictions_df = pd.DataFrame(predictions_list)

# Guardar las predicciones en un archivo CSV
predictions_df.to_csv('predicted_data_with_sentiment.csv', index=False)
print("Predicciones y sentimiento guardados en 'predicted_data_with_sentiment.csv'")

# Comparar predicciones con valores reales
comparison_df = pd.merge(predictions_df, data[['timestamp', 'close']], left_on='date', right_on='timestamp')
comparison_df = comparison_df.drop(columns=['timestamp'])
comparison_df = comparison_df.rename(columns={'close': 'actual'})

# Nueva estrategia de inversión basada en predicción y sentimiento
investment_df = comparison_df.copy()
investment_df['price_direction'] = np.where(investment_df['prediction'].shift(-1) > investment_df['prediction'], 1, -1)
investment_df['sentiment_direction'] = np.where(investment_df['sentiment'] > 0, 1, -1)
investment_df['position'] = np.where(investment_df['price_direction'] == investment_df['sentiment_direction'], investment_df['price_direction'], 0)
investment_df['strategy_returns'] = investment_df['position'] * (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']
investment_df['buy_and_hold_returns'] = (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']


# Calcular métricas de error
mae = mean_absolute_error(comparison_df['actual'], comparison_df['prediction'])
rmse = np.sqrt(mean_squared_error(comparison_df['actual'], comparison_df['prediction']))
r2 = r2_score(comparison_df['actual'], comparison_df['prediction'])
mape = mean_absolute_percentage_error(comparison_df['actual'], comparison_df['prediction'])
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# Dibujar la gráfica con bandas de error
plt.figure(figsize=(14, 7))
plt.plot(comparison_df['date'], comparison_df['actual'], label='Actual Price', color='blue')
plt.plot(comparison_df['date'], comparison_df['prediction'], label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{symbol} Price Prediction vs Actual')
plt.legend()
plt.savefig(f"{symbol.replace('/', '_')}_price_prediction.png")
plt.show()
print(f"Gráfica guardada como '{symbol.replace('/', '_')}_price_prediction.png'")

# Análisis de errores residuales
residuals = comparison_df['actual'] - comparison_df['prediction']
plt.figure(figsize=(14, 7))
plt.plot(comparison_df['date'], residuals, label='Residuals', color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title(f'{symbol} Prediction Residuals')
plt.legend()
plt.savefig(f"{symbol.replace('/', '_')}_residuals.png")
plt.show()
print(f"Gráfica de residuales guardada como '{symbol.replace('/', '_')}_residuals.png'")

# Análisis de correlación
correlation = comparison_df['actual'].corr(comparison_df['prediction'])
print(f'Correlation between actual and predicted prices: {correlation}')

# Nueva estrategia de inversión basada en predicción y sentimiento
investment_df = comparison_df.copy()
investment_df['price_direction'] = np.where(investment_df['prediction'].shift(-1) > investment_df['prediction'], 1, -1)
investment_df['sentiment_direction'] = np.where(investment_df['sentiment'] > 0, 1, -1)
investment_df['position'] = np.where(investment_df['price_direction'] == investment_df['sentiment_direction'], investment_df['price_direction'], 0)
investment_df['strategy_returns'] = investment_df['position'] * (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']
investment_df['buy_and_hold_returns'] = (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']

strategy_cumulative_returns = (investment_df['strategy_returns'] + 1).cumprod() - 1
buy_and_hold_cumulative_returns = (investment_df['buy_and_hold_returns'] + 1).cumprod() - 1

investment_df['peak'] = investment_df['strategy_returns'].cummax()
investment_df['drawdown'] = (investment_df['strategy_returns'] - investment_df['peak']) / (1 + investment_df['peak'])


plt.figure(figsize=(14, 7))
plt.plot(investment_df['date'], strategy_cumulative_returns, label='Strategy Cumulative Returns', color='green')
plt.plot(investment_df['date'], buy_and_hold_cumulative_returns, label='Buy and Hold Cumulative Returns', color='orange')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title(f'{symbol} Investment Strategy vs Buy and Hold')
plt.legend()
plt.savefig(f"{symbol.replace('/', '_')}_investment_strategy.png")
plt.show()
print(f"Gráfica de estrategia de inversión guardada como '{symbol.replace('/', '_')}_investment_strategy.png'")

# Análisis de drawdown
investment_df['drawdown'] = strategy_cumulative_returns.cummax() - strategy_cumulative_returns
investment_df['max_drawdown'] = investment_df['drawdown'].max()

plt.figure(figsize=(14, 7))
plt.plot(investment_df['date'], investment_df['drawdown'], label='Drawdown', color='red')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.title(f'{symbol} Strategy Drawdown')
plt.legend()
plt.savefig(f"{symbol.replace('/', '_')}_drawdown.png")
plt.show()
print(f"Gráfica de drawdown guardada como '{symbol.replace('/', '_')}_drawdown.png'")

# Ratio de Sharpe de la estrategia
risk_free_rate = 0.01  # Asume una tasa libre de riesgo anual del 1%
strategy_returns_daily = investment_df['strategy_returns'].dropna()
excess_returns = strategy_returns_daily - risk_free_rate / 252
sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
print(f'Sharpe Ratio: {sharpe_ratio}')

# Calcular métricas adicionales: Índice de Sortino, Beta y Alfa
# Calcular el Ratio de Sortino
risk_free_rate = 0.01  # Asume una tasa libre de riesgo anual del 1%
strategy_returns_daily = investment_df['strategy_returns'].dropna()
excess_returns = strategy_returns_daily - risk_free_rate / 252

# Manejar posibles NaN o inf en excess_returns
excess_returns = excess_returns.replace([np.inf, -np.inf], np.nan).dropna()

downside_returns = excess_returns[excess_returns < 0]

if len(downside_returns) > 0 and len(excess_returns) > 0:
    sortino_ratio = (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(252)
    print(f'Sortino Ratio: {sortino_ratio:.4f}')
else:
    print('No se puede calcular el Ratio de Sortino debido a datos insuficientes o ausencia de rendimientos negativos.')

# Imprimir información adicional para diagnóstico
print(f'Número de rendimientos totales: {len(strategy_returns_daily)}')
print(f'Número de rendimientos en exceso: {len(excess_returns)}')
print(f'Número de rendimientos negativos: {len(downside_returns)}')
print(f'Media de rendimientos en exceso: {np.mean(excess_returns):.6f}')
print(f'Desviación estándar de rendimientos negativos: {np.std(downside_returns):.6f}')


# Índice de Sortino
downside_returns = strategy_returns_daily[strategy_returns_daily < 0]
sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
print(f'Sortino Ratio: {sortino_ratio}')

# Beta y Alfa
market_returns = investment_df['buy_and_hold_returns'].dropna()
covariance_matrix = np.cov(strategy_returns_daily, market_returns)
beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
alpha = np.mean(strategy_returns_daily) - beta * np.mean(market_returns)
print(f'Beta: {beta}')
print(f'Alpha: {alpha}')

# Validación cruzada
tscv = TimeSeriesSplit(n_splits=5)
cross_val_scores = []
for train_index, test_index in tscv.split(data):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    
    train.loc[:, 'close_normalized'] = train['close'].rolling(window=sequence_length, min_periods=1).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)), raw=True)
    test.loc[:, 'close_normalized'] = test['close'].rolling(window=sequence_length, min_periods=1).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)), raw=True)
    
    predictions_cv = []
    for i in range(len(test) - sequence_length):
        input_window = train['close_normalized'].values[-sequence_length+i:]
        input_window = np.append(input_window, test['close_normalized'].values[:i+1])
        input_window = np.array(input_window[-sequence_length:]).astype(np.float32)
        input_window = np.expand_dims(input_window, axis=0)
        input_window = np.expand_dims(input_window, axis=2)
        
        output = ort_session.run(None, {input_name: input_window})
        prediction = output[0][0][0]
        prediction = prediction * (max_close_window - min_close_window) + min_close_window
        predictions_cv.append(prediction)
    
    actuals_cv = test['close'].values[sequence_length:]
    mae_cv = mean_absolute_error(actuals_cv, predictions_cv)
    cross_val_scores.append(mae_cv)

print(f'Cross-Validation MAE: {np.mean(cross_val_scores)} ± {np.std(cross_val_scores)}')

# Comparación con modelo de media móvil simple (SMA)
data['SMA'] = data['close'].rolling(window=sequence_length).mean()

# Predicciones del modelo de media móvil
data = data.dropna()
sma_predictions = data['SMA'].values
sma_actuals = data['close'].values

sma_mae = mean_absolute_error(sma_actuals, sma_predictions)
sma_rmse = np.sqrt(mean_squared_error(sma_actuals, sma_predictions))
sma_r2 = r2_score(sma_actuals, sma_predictions)
print(f'SMA Mean Absolute Error (MAE): {sma_mae}')
print(f'SMA Mean Absolute Error (MAE): {sma_mae}')
print(f'SMA Root Mean Squared Error (RMSE): {sma_rmse}')
print(f'SMA R-squared (R2): {sma_r2}')

plt.figure(figsize=(14, 7))
plt.plot(data['timestamp'], data['close'], label='Actual Price', color='blue')
plt.plot(data['timestamp'], data['SMA'], label='SMA Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{symbol} SMA Price Prediction vs Actual')
plt.legend()
plt.savefig(f"{symbol.replace('/', '_')}_sma_price_prediction.png")
plt.show()
print(f"Gráfica de predicción SMA guardada como '{symbol.replace('/', '_')}_sma_price_prediction.png'")

# Crear una gráfica que muestre el sentimiento, el precio real y la predicción
fig, ax1 = plt.subplots(figsize=(16, 8))

# Eje para el precio y la predicción
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color='tab:blue')
ax1.plot(comparison_df['date'], comparison_df['actual'], label='Actual Price', color='tab:blue')
ax1.plot(comparison_df['date'], comparison_df['prediction'], label='Predicted Price', color='tab:orange')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Crear un segundo eje para el sentimiento
ax2 = ax1.twinx()
ax2.set_ylabel('Sentiment', color='tab:green')
ax2.plot(comparison_df['date'], comparison_df['sentiment'], label='Sentiment', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

# Añadir leyendas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title(f'{symbol} Price, Prediction, and Sentiment Over Time')
plt.savefig(f"{symbol.replace('/', '_')}_price_prediction_sentiment.png")
plt.show()
print(f"Gráfica de precio, predicción y sentimiento guardada como '{symbol.replace('/', '_')}_price_prediction_sentiment.png'")

plt.figure(figsize=(14, 7))
plt.plot(investment_df['date'], investment_df['drawdown'], label='Drawdown', color='red')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.title(f'{symbol} Strategy Drawdown')
plt.legend()
plt.savefig(f"{symbol.replace('/', '_')}_drawdown.png")
plt.show()
print(f"Gráfica de drawdown guardada como '{symbol.replace('/', '_')}_drawdown.png'")

# Print maximum drawdown
print(f"Maximum Drawdown: {investment_df['drawdown'].min():.2%}")
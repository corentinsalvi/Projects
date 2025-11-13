#####################################################################################################
#                                                                                                   #
#                Global Analysis and Prediction of the Bitcoin (15min) - 2018 à 2025                #
#                                                                                                   #
#  Corentin SALVI                                                                                   #
#  11/13/2025                                                                                       # 
#####################################################################################################

# region Libraries Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# endregion

# region Data Loading Process
def load_bitcoin_data():
    """
    Load the Bitcoin 15min dataset
    """
    print("Chargement du dataset")
    df = pd.read_csv('btc_15m_data_2018_to_2025.csv')
    
    # Convert date columns
    df['Open time'] = pd.to_datetime(df['Open time'])
    df['Close time'] = pd.to_datetime(df['Close time'])
    
    df.set_index('Open time', inplace=True)
    df.sort_index(inplace=True)
    
    return df
# endregion

# region Data Loading and Overview
btc_df = load_bitcoin_data()
print("Shape du dataset:", btc_df.shape)
print("\nPériode couverte:", btc_df.index.min(), "à", btc_df.index.max())
print("\nPremières lignes:")
print(btc_df.head())
print("\nNoms des colonnes disponibles:")
print(btc_df.columns.tolist())
print("Valeurs manquantes par colonne :")
print(btc_df.isnull().sum())
# endregion 


# region Data Visualization
plt.figure(figsize=(15, 12))
# Closing price over the entire period
plt.subplot(3, 2, 1)
plt.plot(btc_df.index, btc_df['Close'], color='orange', linewidth=0.8)
plt.title('Prix Bitcoin (15min) - 2018-2025')
plt.xlabel('Date')
plt.ylabel('Prix (USD)')
plt.grid(True)
# Trade volume
plt.subplot(3, 2, 2)
plt.plot(btc_df.index, btc_df['Volume'], color='blue', alpha=0.7, linewidth=0.5)
plt.title('Volume des échanges (15min)')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
# Returns within 15 minutes
btc_df['Returns_15min'] = btc_df['Close'].pct_change()
plt.subplot(3, 2, 3)
plt.plot(btc_df.index, btc_df['Returns_15min'] * 100, color='green', linewidth=0.3)
plt.title('Returns sur 15min (%)')
plt.xlabel('Date')
plt.ylabel('Return (%)')
plt.grid(True)
# Price distribution
plt.subplot(3, 2, 4)
plt.hist(btc_df['Close'].dropna(), bins=100, color='purple', alpha=0.7)
plt.title('Distribution des prix de clôture')
plt.xlabel('Prix (USD)')
plt.ylabel('Fréquence')
plt.grid(True)
# Volatility (High - Low)
btc_df['Volatility'] = (btc_df['High'] - btc_df['Low']) / btc_df['Open'] * 100
plt.subplot(3, 2, 5)
plt.plot(btc_df.index, btc_df['Volatility'], color='red', linewidth=0.5, alpha=0.7)
plt.title('Volatilité (%) par période 15min')
plt.xlabel('Date')
plt.ylabel('Volatilité (%)')
plt.grid(True)
# Correlation between main variables
plt.subplot(3, 2, 6)
corr_matrix = btc_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns_15min']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matrice de Corrélation')
plt.tight_layout()
plt.savefig('analyse_globale_bitcoin.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: analyse_globale_bitcoin.png")
plt.show()
# endregion 

# region Feature Engineering
def create_custom_features(df):
    """
    Transformation of raw data into technical indices and features adapted for 15min data
    """
    df = df.copy()
    
    # Lags (OK)
    for lag in [1, 4, 5, 12, 13]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    # Moyennes mobiles AVEC décalage
    for window in [12, 24]:
        df[f'MA_{window}'] = df['Close_Lag_1'].rolling(window=window).mean() 
        df[f'Volume_MA_{window}'] = df['Volume_Lag_1'].rolling(window=window).mean()
    
    # Momentum AVEC décalage  
    df['Momentum_4'] = df['Close_Lag_1'] - df['Close_Lag_5'] 
    df['Momentum_12'] = df['Close_Lag_1'] - df['Close_Lag_13']
    
    # RSI sécurisé
    close_lag_1 = df['Close'].shift(1)
    delta = close_lag_1.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD sécurisé
    exp1 = df['Close_Lag_1'].ewm(span=12).mean()
    exp2 = df['Close_Lag_1'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    
    # Bollinger Bands sécurisé
    df['BB_Middle'] = df['Close_Lag_1'].rolling(window=20).mean()
    bb_std = df['Close_Lag_1'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Position'] = (df['Close_Lag_1'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume ratio sécurisé
    df['Volume_MA_12'] = df['Volume_Lag_1'].rolling(window=12).mean()
    df['Volume_Ratio'] = df['Volume_Lag_1'] / df['Volume_MA_12']
    
    # Time features (OK)
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    return df
# endregion 

# region Feature Engineering Application
print("Création des features techniques")
btc_df = create_custom_features(btc_df)
corr_15min = btc_df['Close'].corr(btc_df['Close'].shift(1))
print(f"Nouvelle corrélation 15min: {corr_15min:.4f}")
print(f"Shape après feature engineering: {btc_df.shape}")
print(f"Nombre de features créées: {len([col for col in btc_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns_15min', 'Volatility']])}")
# endregion 

# region Data preparation for machine learning
def prepare_custom_data(df, target_col='Close', forecast_periods=4):
    """
    Prepare the data for prediction
    """
    df = df.copy()   
    # Create the target variable (future price)
    df['Target'] = df[target_col].shift(-forecast_periods)
    
    # Remove rows with NaN
    df_clean = df.dropna()                                                                                                                                                                 
    
    # Select features (exclude non-numeric columns and the target)
    exclude_cols = ['Target', 'Open time', 'Close time', 'Ignore','Close', 'Open', 'High', 'Low', 'Volume', 'Returns_15min', 'Volatility']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols and df_clean[col].dtype in ['float64', 'int64']]
    
    # 
    X = df_clean[feature_cols]
    y = df_clean['Target']
    
    return X, y, feature_cols

X, y, feature_cols = prepare_custom_data(btc_df, forecast_periods=4)


print(f"Features disponibles: {len(feature_cols)}")
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
# endregion 

# region Split data according to time
def time_series_split_custom(X, y, test_size=0.2):
    """
    Chronological separation of data into training and testing sets
    """
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = time_series_split_custom(X, y, test_size=0.2)
# endregion

# Displaying datasets for training and testing
print(f"Training set: {X_train.shape[0]} périodes")
print(f"Test set: {X_test.shape[0]} périodes")
print(f"Date de début test: {X_test.index[0]}")
print(f"Date de fin test: {X_test.index[-1]}")

# Feature normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# region Model training and evaluation
def evaluate_custom_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Model training and performance evaluation with 8 different metrics 
    """
    # Training
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluation metrics
    metrics = {
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train_MAE': mean_absolute_error(y_train, y_pred_train),
        'Test_MAE': mean_absolute_error(y_test, y_pred_test),
        'Train_R2': r2_score(y_train, y_pred_train),
        'Test_R2': r2_score(y_test, y_pred_test),
        'Train_MAPE': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100,
        'Test_MAPE': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    }
    
    print(f"\n{model_name} Performance:")
    print(f"Train RMSE: ${metrics['Train_RMSE']:.2f}")
    print(f"Test RMSE: ${metrics['Test_RMSE']:.2f}")
    print(f"Train MAE: ${metrics['Train_MAE']:.2f}")
    print(f"Test MAE: ${metrics['Test_MAE']:.2f}")
    print(f"Train R²: {metrics['Train_R2']:.4f}")
    print(f"Test R²: {metrics['Test_R2']:.4f}")
    print(f"Train MAPE: {metrics['Train_MAPE']:.2f}%")
    print(f"Test MAPE: {metrics['Test_MAPE']:.2f}%")
    
    return model, y_pred_test, metrics
# endregion 

# region Model initialization
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,             
        min_samples_split=50,      
        min_samples_leaf=25,     
        max_features=0.5,          
        random_state=42,
        n_jobs=-1
        )
}
# endregion

# region Training and assessment
results = {}
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Entraînement de {name}...")
    trained_model, predictions, metrics = evaluate_custom_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test, name
    )
    results[name] = {
        'model': trained_model,
        'predictions': predictions,
        'metrics': metrics
    }
# endregion 

# region Results visualization
plt.figure(figsize=(16, 12))

# Real vs Predicted Price Comparison
plt.subplot(3, 2, 1)
sample_size = min(500, len(y_test)) 
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

plt.scatter(y_test.values[sample_indices], 
           results['Random Forest']['predictions'][sample_indices], 
           alpha=0.6, color='blue', label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Prix Réel (USD)')
plt.ylabel('Prix Prédit (USD)')
plt.title('Prix Réel vs Prédit')
plt.legend()
plt.grid(True)

# Time Series of Predictions
plt.subplot(3, 2, 2)
plot_size = min(200, len(y_test))
plt.plot(y_test.index[:plot_size], y_test.values[:plot_size], 
         label='Prix Réel', color='black', linewidth=2)
for name, result in results.items():
    plt.plot(y_test.index[:plot_size], result['predictions'][:plot_size], 
             label=f'{name} Prédiction', linestyle='--', alpha=0.8)
plt.title('Prédictions vs Réel (extrait)')
plt.xlabel('Date')
plt.ylabel('Prix (USD)')
plt.legend()
plt.grid(True)

# Error Distribution
plt.subplot(3, 2, 3)
for name, result in results.items():
    errors = y_test.values - result['predictions']
    plt.hist(errors, bins=50, alpha=0.6, label=f'{name} Erreurs')
plt.title('Distribution des Erreurs de Prédiction')
plt.xlabel('Erreur (USD)')
plt.ylabel('Fréquence')
plt.legend()
plt.grid(True)

# Feature Importance
plt.subplot(3, 2, 4)
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
    plt.title('Top 10 Features les plus importantes')
    plt.xlabel('Score d\'importance')

# region Model Performance
plt.subplot(3, 2, 5)
metrics_compare = ['Test_RMSE', 'Test_MAE', 'Test_MAPE']
model_names = list(results.keys())
x_pos = np.arange(len(model_names))

for i, metric in enumerate(metrics_compare):
    values = [results[name]['metrics'][metric] for name in model_names]
    plt.bar(x_pos + i*0.25, values, width=0.25, 
            label=metric.replace('Test_', ''), alpha=0.8)

plt.xlabel('Modèles')
plt.ylabel('Valeur des Métriques')
plt.title('Comparaison des Performances')
plt.xticks(x_pos + 0.25, model_names)
plt.legend()
plt.grid(True, alpha=0.3)

# Cumulative Errors
plt.subplot(3, 2, 6)
for name, result in results.items():
    cumulative_errors = np.cumsum(np.abs(y_test.values - result['predictions']))
    plt.plot(y_test.index, cumulative_errors, label=f'{name} Erreurs Cumulatives')
plt.title('Erreurs Absolues Cumulatives')
plt.xlabel('Date')
plt.ylabel('Erreur Cumulative (USD)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('resultats_performance_model.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: resultats_performance_model.png")
plt.show()
# endregion 

# region Trading strategy simulation
def simulate_15min_trading(actual_prices, predicted_prices, initial_capital=10000):
    """
    This function simulates a trading strategy that buys when the predicted price exceeds the current price by 0.1% and sells when the opposite occurs, then compares the performance obtained with a simple buy-and-hold strategy.
    """
    capital = initial_capital
    position = 0 
    trades = []
    portfolio_values = []
    
    for i in range(1, len(actual_prices)):
        current_price = actual_prices[i-1]
        predicted_price = predicted_prices[i]
        
        # Prediction-based signal (buy if predicted price > current price + threshold)
        price_threshold = current_price * 0.001  # Threshold of 0.1%
        
        if position == 0 and predicted_price > current_price + price_threshold:
            # Buy
            position = capital / current_price
            capital = 0
            trades.append(('BUY', i, current_price))
        
        elif position > 0 and predicted_price < current_price - price_threshold:
            # Sell
            capital = position * current_price
            position = 0
            trades.append(('SELL', i, current_price))
        
        # Portfolio value
        portfolio_value = capital + (position * actual_prices[i] if position > 0 else 0)
        portfolio_values.append(portfolio_value)
    
    # Final value
    final_value = capital + (position * actual_prices[-1] if position > 0 else 0)
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Comparison with buy & hold
    buy_hold_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'trades': trades,
        'portfolio_values': portfolio_values
    }
# endregion

# region Testing the strategy with Random Forest
if 'Random Forest' in results:
    strategy_results = simulate_15min_trading(y_test.values, results['Random Forest']['predictions'])
    
    print("Resultats de la stratégie de trading")
    print("-"*50)
    print(f"Capital initial: $10,000")
    print(f"Valeur finale: ${strategy_results['final_value']:.2f}")
    print(f"Rendement stratégie: {strategy_results['total_return']:.2f}%")
    print(f"Rendement Buy & Hold: {strategy_results['buy_hold_return']:.2f}%")
    print(f"Nombre de trades: {len(strategy_results['trades'])}")
    print(f"Performance vs Buy & Hold: {strategy_results['total_return'] - strategy_results['buy_hold_return']:.2f}%")
#endregion 

# # region Analyse de robustesse par période
# print("Analyse de la robustesse du modele")
# print("-"*50)
# # Diviser le test set en périodes
# periods = 4
# period_size = len(y_test) // periods

# for i in range(periods):
#     start_idx = i * period_size
#     end_idx = (i + 1) * period_size if i < periods - 1 else len(y_test)
    
#     period_actual = y_test.iloc[start_idx:end_idx]
#     period_dates = f"{period_actual.index[0].date()} to {period_actual.index[-1].date()}"
    
#     print(f"\nPériode {i+1} ({period_dates}):")
    
#     for name, result in results.items():
#         period_pred = result['predictions'][start_idx:end_idx]
#         period_rmse = np.sqrt(mean_squared_error(period_actual, period_pred))
#         period_r2 = r2_score(period_actual, period_pred)
#         print(f"  {name}: RMSE = ${period_rmse:.2f}, R² = {period_r2:.4f}")
# # endregion


print("Synthese finale")
print("-"*60)

print("\nPerformances obtenues:")
for name, result in results.items():
    metrics = result['metrics']
    print(f"   {name}:")
    print(f"  - Erreur moyenne: ${metrics['Test_MAE']:.2f}")
    print(f"  - Précision: {metrics['Test_R2']:.4f} R²")

print("\n Ameliorations possibles:")
print("   - Ajouter des features externes (sentiment...)")
print("   - Essayer d'autres modeles plus avances comme le Gradient Boosting ou les RNN")

print("\n Limitations:")
print("   - Les marchés crypto sont très volatils")
print("   - Les performances passées ne garantissent pas les futures")


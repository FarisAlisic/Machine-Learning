import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression

def load_and_train_model():
    df = pd.read_csv('/Users/farisalicic/Desktop/programming/ML/phoneprice.csv')
    df = pd.concat([df, pd.get_dummies(df['Brand'], prefix='Brand')], axis=1)
    df.drop(columns=['Brand'], inplace=True)

    X = df.drop('Price', axis=1)
    y = df['Price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

def get_user_input(feature_names):
    print("\nPlease enter the phone specifications:")

    # Get brand input
    brand_columns = [col for col in feature_names if col.startswith('Brand_')]
    brands = [col.replace('Brand_', '') for col in brand_columns]
    
    print("\nAvailable brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
    
    while True:
        try:
            brand_idx = int(input("\nSelect brand number: ")) - 1
            if 0 <= brand_idx < len(brands):
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    user_input = np.zeros(len(feature_names))
    
    # Set the selected brand to 1
    brand_col_idx = feature_names.get_loc(f"Brand_{brands[brand_idx]}")
    user_input[brand_col_idx] = 1
    
    # Get other specifications
    specs = {
        'Storage': 'Storage (GB): ',
        'RAM': 'RAM (GB): ',
        'Screen Size': 'Screen Size (inches): ',
        'Camera': 'Total Camera MP: ',
        'Battery Capacity': 'Battery Capacity (mAh): '
    }
    
    for spec, prompt in specs.items():
        while True:
            try:
                value = float(input(prompt))
                if value > 0:
                    user_input[feature_names.get_loc(spec)] = value
                    break
                print("Please enter a positive value.")
            except ValueError:
                print("Please enter a valid number.")
    
    return user_input

def main():
    model, scaler, feature_names = load_and_train_model()
    
    while True:
        user_input = get_user_input(feature_names)
        user_input_scaled = scaler.transform(user_input.reshape(1, -1))
        predicted_price = model.predict(user_input_scaled)[0]
        
        print(f"\nPredicted Price: ${predicted_price:.2f}")
        
        again = input("\nWould you like to predict another price? (yes/no): ").lower()
        if again != 'yes':
            break
    
    print("\nThank you for using the phone price predictor!")

if __name__ == "__main__":
    main()
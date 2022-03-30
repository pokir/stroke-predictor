import torch


while True:
    existing_model_path = input('Model path: ')
    try:
        model = torch.load(existing_model_path)
        break
    except:
        print('Invalid model file')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

print(f'Model: {model}')

print()


def predict(X):
    with torch.no_grad():
        return model(X)


# gender: Male, Female, Other
# age
# hypertension
# heart disease
# ever married: No, Yes
# work: children, Govt_jov, Never worked, Private, Self-employed
# residence: Rural, Urban
# glucose level
# bmi
# smoking: formerly smoked, never smoked, smokes, Unknown

X = torch.tensor([0, 17, 0, 0, 0, 2, 1, 30, 20, 1], dtype=torch.float32)

pred = predict(X)

print(f'You have a {pred.item() * 100:.2f}% chance of getting a stroke')

import pandas as pd
import statistics
import matplotlib.pyplot as plot

file_path = "C:/Users/ashwi/Desktop/Sem 4/ML/Lab_ML/Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day_name()

#1
price_data = df["Price"]  # Column D (Price)
mean = statistics.mean(price_data)
variance = statistics.variance(price_data)
print(f"Mean of Price Data: {mean:.2f}")
print(f"Variance of Price Data: {variance:.2f}")

#2
wednesday_data = df[df["Day"] == "Wednesday"]["Price"]
smean = statistics.mean(wednesday_data)
print(f"Mean Price on Wednesdays: {smean:.2f}")
print(f"Difference from Population Mean: {abs(smean - mean):.2f}")

#3
april_data = df[df["Date"].dt.month == 4]["Price"]
mean_april = statistics.mean(april_data)
print(f"Mean Price in April: {mean_april:.2f}")
print(f"Difference from Population Mean: {abs(mean_april - mean):.2f}")

#4
lossprob = (df["Chg%"].apply(lambda x: x < 0).sum()) / len(df)
print(f"Probability of making a loss: {lossprob:.2f}")

#5
prob_wed = (df[(df["Day"] == "Wednesday") & (df["Chg%"] > 0)].shape[0]) / (df[df["Day"] == "Wednesday"].shape[0])
print(f"Probability of making a profit on Wednesday: {prob_wed:.2f}")

#6
condprob = prob_wed  # Same as previous calculation since
print(f"Conditional probability of profit given today is Wednesday: {condprob:.2f}")

#7
plot.figure(figsize=(10, 5))
plot.scatter(df["Day"], df["Chg%"], alpha=0.5, color="blue")
plot.xlabel("Day of the Week")
plot.ylabel("Chg%")
plot.title("Stock Percentage Change vs. Day of the Week")
plot.xticks(rotation=45)
plot.grid(True)
plot.show()

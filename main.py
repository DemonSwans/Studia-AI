from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Mam nadzieję że ta ilość komentarzy wystarczy gdyż i pan i ja dokładnie wiemy
# co użyte przeze mnie funkcje robią a pisanie zbędnych komentarzy które mówią że
# białe jest białe a czarne jest czarne jest niepotrzebne

# Wczytywanie danych
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
#Wypisanie informacji które pan chciał ale przed obróbką
print("Przed Obróbką Danych")
print(f"Liczba Rzędów: {len(df.axes[0])}")
print(f"Liczba Kolumn: {len(df.axes[1])}")
print(f"Typy Danych:\n{df.dtypes}")

"""
Przygotowanie danych
Np. Przypisanie do wartości pisemnych odpowiedników liczbowych czy konwersja daty na datetime
"""
#Zanim zamienie typy pogody na odpowiedniki liczbowe zrobię wykres kołołowy pokazujący %-owe występowanie typów pogody
counted_weather = []
labels = df.weather_main.unique()
for i in labels:
    counted_weather.append(df.weather_main.value_counts()[i])
plt.pie(counted_weather, labels=labels, autopct='%1.1f%%')
plt.title('Wykres Kołowy % Typu Pogody')
plt.show()
#To jest słownik
#      |
#      v
funday = {'None':0,
'Columbus Day':1,
'Veterans Day':2,
'Thanksgiving Day':3,
'Christmas Day':4,
'New Years Day':5,
'Washingtons Birthday':6,
'Memorial Day':7,
'Independence Day':8,
'State Fair':9,
'Labor Day':10,
'Martin Luther King Jr Day':11,}
# W skrócie ten kod zamienia wartości z kolumny "holiday" na wartości odpowiadające im z słownika "funday"
df.holiday = [funday[item] for item in df.holiday]
#----------------------------------------------------------------------
#To jest słownik
#      |
#      v
weather = {'Clouds':1 ,
           'Clear':2 ,
           'Rain':3,
           'Drizzle':4,
           'Mist':5,
           'Haze':6,
           'Fog':7,
           'Thunderstorm':8,
           'Snow':9,
           'Squall':10,
           'Smoke':11}
# W skrócie ten kod zamienia wartości z kolumny "holiday" na wartości odpowiadające im z słownika "weather".
df.weather_main = [weather[item] for item in df.weather_main]
#----------------------------------------------------------------------
#To jest słownik
#      |
#      v
weather_desc = {'scattered clouds':0,
'broken clouds':1,
'overcast clouds':2,
'sky is clear':3,
'few clouds':4,
'light rain':5,
'light intensity drizzle':6,
'mist':7,
'haze':8,
'fog':9,
'proximity shower rain':10,
'drizzle':11,
'moderate rain':12,
'heavy intensity rain':13,
'proximity thunderstorm':14,
'thunderstorm with light rain':15,
'proximity thunderstorm with rain':16,
'heavy snow':17,
'heavy intensity drizzle':18,
'snow':19,
'thunderstorm with heavy rain':20,
'freezing rain':21,
'shower snow':22,
'light rain and snow':23,
'light intensity shower rain':24,
'SQUALLS':25,
'thunderstorm with rain':26,
'proximity thunderstorm with drizzle':27,
'thunderstorm':28,
'Sky is Clear':29,
'very heavy rain':30,
'thunderstorm with light drizzle':31,
'light snow':32,
'thunderstorm with drizzle':33,
'smoke':34,
'shower drizzle':35,
'light shower snow':36,
'sleet':37,}\
# W skrócie ten kod zamienia wartości z kolumny "weather_description" na wartości odpowiadające im z słownika "weather_desc".
df.weather_description = [weather_desc[item] for item in df.weather_description]
#----------------------------------------------------------------------
# W skrócie używając funkcji pandas "to_datetime" do konwersji na typ danych datetime z których zapisuję miesiąc,dzień i godzinę a nastepnie usuwam kolumne "date_time"
df['date_time'] = pd.to_datetime(df['date_time'])
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour
df.drop("date_time", axis=1, inplace=True)
#----------------------------------------------------------------------
#Wypisanie informacji które pan chciał ale po obróbce
print("-------------------------------------")
print("Po Obróbce Danych")
print(f"Liczba Rzędów: {len(df.axes[0])}")
print(f"Liczba Kolumn: {len(df.axes[1])}")
print(f"Typy Danych:\n{df.dtypes}")
print("-------------------------------------")
#To jest lista z danymi typu string
#      |
#      v
features = ["holiday","temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "weather_description", 'month','day','hour']
#Definiuję dataframe z kolumnami które mają wykazać
x = df[features]
#To jest string
#      |
#      v
labels = "traffic_volume"
y = df[labels]

# Dziele dane na zbiór treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

# Towrzę i trenuje model drzewa decyzyjnego
regressor = DecisionTreeRegressor(splitter="best",random_state=0)
regressor.fit(x_train.values, y_train.values)

# Ocenianie modelu na danych testowych
score = regressor.score(x_test.values, y_test.values)
print("Accuracy:", f'{int(score*100)}%')
print("-------------------------------------")
# Przykład działania te dane zostały wcześniej usunięte z pliku CSV
print("Dane: 0,290.37,0.0,0.0,90,1,2,10,9,17\nIlość Pasażerów: 6127")
print("-------------------------------------")
print(f"Predykcja: {int(regressor.predict([[0,290.37,0.0,0.0,90,1,2,10,9,17]])[0])}")

#Tworzę histogram średniej ilości pasażerów z podziałem na godziny
mean_traffic_by_hour = df.groupby('hour').mean()['traffic_volume']
mean_traffic_by_hour.plot(kind='bar', color='red',edgecolor='black')
plt.title('Histogram średniej ilości pasażerów na godzinę')
plt.xlabel('Godzina')
plt.ylabel('Średnia ilości pasażerów')
plt.show()
#---------------------------------------------
#Tworzę wykres liniowy średniej ilości pasażerów z podziałem na godziny
hourly_avg = df.groupby("hour").mean()["traffic_volume"]
hourly_avg.plot(kind="line")
plt.xlabel("Godzina")
plt.ylabel("Średnia ilość ruchu")
plt.title("Średni ruch w różnych godzinach dnia")
plt.show()
#---------------------------------------------
#Tworzę wykres kołowy średniej ilości pasażerów z podziałem na dni miesiąca
weekday_proportions = df["day"].value_counts() / len(df)
weekday_proportions.plot(kind="pie")
plt.title("Proporcje ruchu w różne dni miesiąca")
plt.show()
#---------------------------------------------
#Tworzę wykres Violinowy ilości pasażerów
sns.violinplot(df["traffic_volume"])
plt.ylabel("Ilość Pasażerów")
plt.title("Ilość Pasażerów Wykres Violinowy")
plt.show()
#---------------------------------------------
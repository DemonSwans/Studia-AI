#Plik pomocniczy bo byłem zbyt leniwy by samemu dzielić to do słownika
st = "'Columbus Day' 'Veterans Day' 'Thanksgiving Day' 'Christmas Day' 'New Years Day' 'Washingtons Birthday' 'Memorial Day' 'Independence Day' 'State Fair' 'Labor Day' 'Martin Luther King Jr Day'"
d = st.split("'")
for i in range(len(d)):
    if i % 2 == 1:
        print(f"'{d[i]}':{int((i-1)/2)+1},")


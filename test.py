import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle

## data

allheroes =['Abaddon', 'Alchemist', 'Ancient Apparition', 'Anti-Mage', 'Arc Warden', 'Axe', 'Bane', 'Batrider', 'Beastmaster', 'Bloodseeker', 'Bounty Hunter', 'Brewmaster', 'Bristleback', 'Broodmother', 'Centaur Warrunner', 'Chaos Knight', 'Chen', 'Clinkz', 'Clockwerk', 'Crystal Maiden', 'Dark Seer', 'Dark Willow', 'Dawnbreaker', 'Dazzle', 'Death Prophet', 'Disruptor', 'Doom', 'Dragon Knight', 'Drow Ranger', 'Earth Spirit', 'Earthshaker', 'Elder Titan', 'Ember Spirit', 'Enchantress', 'Enigma', 'Faceless Void', 'Grimstroke', 'Gyrocopter', 'Hoodwink', 'Huskar', 'Invoker', 'Io', 'Jakiro', 'Juggernaut', 'Keeper of the Light', 'Kunkka', 'Legion Commander', 'Leshrac', 'Lich', 'Lifestealer', 'Lina', 'Lion', 'Lone Druid', 'Luna', 'Lycan', 'Magnus', 'Marci', 'Mars', 'Medusa', 'Meepo', 'Mirana', 'Monkey King', 'Morphling', 'Naga Siren', "Nature's Prophet", 'Necrophos', 'Night Stalker', 'Nyx Assassin', 'Ogre Magi', 'Omniknight', 'Oracle', 'Outworld Devourer', 'Pangolier', 'Phantom Assassin', 'Phantom Lancer', 'Phoenix', 'Primal Beast', 'Puck', 'Pudge', 'Pugna', 'Queen of Pain', 'Razor', 'Riki', 'Rubick', 'Sand King', 'Shadow Demon', 'Shadow Fiend', 'Shadow Shaman', 'Silencer', 'Skywrath Mage', 'Slardar', 'Slark', 'Snapfire', 'Sniper', 'Spectre', 'Spirit Breaker', 'Storm Spirit', 'Sven', 'Techies', 'Templar Assassin', 'Terrorblade', 'Tidehunter', 'Timbersaw', 'Tinker', 'Tiny', 'Treant Protector', 'Troll Warlord', 'Tusk', 'Underlord', 'Undying', 'Ursa', 'Vengeful Spirit', 'Venomancer', 'Viper', 'Visage', 'Void Spirit', 'Warlock', 'Weaver', 'Windranger', 'Winter Wyvern', 'Witch Doctor', 'Wraith King', 'Zeus']

alg = ['Gaussian', 'Multinomial', 'Bernoulli', 'Complementary']

df = pd.read_csv('global.csv')

df_proc = pd.read_csv('finaldata.csv')

gaussian = pickle.load(open('nb_gaussian.pkl', 'rb'))
multi = pickle.load(open('nb_multinomial.pkl', 'rb'))
bern = pickle.load(open('nb_bernoulli.pkl', 'rb'))
comp = pickle.load(open('nb_complementary.pkl', 'rb'))

##layout

st.write('Rifky Ariya Pratama - A11.2020.12628')
st.markdown(''' ## Prediksi hasil match berdasarkan hero yang di pilih di Dota 2 ''')
st.markdown(''' Dota adalah sebuah game ber genre MOBA dengan platform PC yang sangat populer.  
Di sebuah game Dota, terdapat 2 tim yang berlawanan, yaitu Radiant dan Dire. Masing masing team mempunyai 5 buah player.
Dota mempunyai banyak hal yang dapat mempengaruhi hasil match, seperti skill, strategi, item, timing dan sebagainya.

Namun di tugas ini, kita hanya akan membahas hero yang di pilih oleh satu sisi.
Dota 2 mempunyai total hero sebanyak 123 hero.
Satu tim akan memilih 5 hero untuk dimainkan, dan kita akan menghitung presentase kemenangan tim tersebut berdasarkan hero yang di pilih.

Kita mempunyai data 10.000 match dari Dota 2 yang berlangsung di seluruh penjuru dunia.
Match tersebut tersebar dari beberapa patch yang berbeda, dan juga berbeda tahun.
Di setiap patch, ada hero baru, item baru, dan juga nerf hero atau item.
Perbedaan antar patch ini akan mempengaruhi prediksi algoritma kita.

Data dibawah ini diambil langsung dari Dota 2 API. ''')

st.write(df)

st.markdown(''' Setelah dilakukan proses data cleaning dan data processing antara lain:  
            - Menghapus kolom yang tidak diperlukan  
            - Menambahkan 4 kolom baru untuk menampung 5 jumlah hero  
            - Membersihkan karakter yang tidak dibutuhkan di string hero  
            - Mengubah nama hero menjadi hero id  
        
Berikut adalah contoh data yang sudah di proses: ''')

st.write(df_proc)

st.markdown(''' ### Masukkan hero yang di pilih untuk melakukan prediksi: ''')

algorithm = st.selectbox('Pilih Algoritma', alg)
heroes = st.multiselect('Pilih Hero', allheroes, max_selections=5)
    
preds =st.button('Predict')
    
if preds:
    chosenheroes = []
    for hero in heroes:
        chosenheroes.append(allheroes.index(hero))
    
    if algorithm == 'Gaussian':
        prediction = gaussian.predict([chosenheroes])
        if prediction == 1:
            prediction = 'Win - Accuracy 59%'
        else :
            prediction = 'Lose - Accuracy 59%'
        st.write(prediction)
    elif algorithm == 'Multinomial':
        prediction = multi.predict([chosenheroes])
        if prediction == 1:
            prediction = 'Win - Accuracy 54%'
        else :
            prediction = 'Lose - Accuracy 54%'
        st.write(prediction)
    elif algorithm == 'Bernoulli':
        prediction = bern.predict([chosenheroes])
        if prediction == 1:
            prediction = 'Win - Accuracy 59%'
        else :
            prediction = 'Lose - Accuracy 59%'
        st.write(prediction)
    elif algorithm == 'Complementary':
        prediction = comp.predict([chosenheroes])
        if prediction == 1:
            prediction = 'Win - Accuracy 52%'
        else :
            prediction = 'Lose - Accuracy 52%'
        st.write(prediction)
else:
    st.write('')
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
model = load_model("word_predictor.h5")

# Sample large paragraph
paragraph = """
The Indian men's cricket team is one of the most successful and followed teams in the world. Over the decades, India has produced some of the finest cricketers to ever play the game — legends like Sachin Tendulkar, MS Dhoni, Rahul Dravid, Virender Sehwag, Anil Kumble, and now modern-day stars like Virat Kohli, Rohit Sharma, Jasprit Bumrah, and Shubman Gill.

Virat Kohli is among the greatest batsmen of the 21st century. Known for his aggressive approach, fitness, and chase mastery, Kohli has won numerous matches for India across all formats. He has over 25,000 international runs and is the fastest to many milestones. One of his iconic innings came during the 2022 T20 World Cup against Pakistan, where he scored a memorable unbeaten 82* at the Melbourne Cricket Ground, leading India to a miraculous last-ball victory.

Rohit Sharma, the current captain of the Indian team in all formats, is admired for his elegant stroke play and calm leadership. He holds the world record for the highest individual score in ODI cricket — a stunning 264 runs. Rohit has scored three double centuries in ODIs, more than any other player in history. As captain of Mumbai Indians in the IPL, he led the franchise to five titles.

Jasprit Bumrah, India's fast-bowling spearhead, is known for his lethal yorkers, control, and temperament in pressure situations. His unorthodox action and ability to bowl in the death overs make him one of the top bowlers in the world. He played a key role in India’s Test series win in Australia in 2020-21 and has delivered in World Cups and IPL alike.

Ravindra Jadeja, one of the world’s finest all-rounders, contributes with bat, ball, and especially in the field. His bullet throws, sharp reflexes, and ability to take wickets and hit quick runs make him a complete package. In the 2019 World Cup semifinal against New Zealand, he scored 77 and almost pulled off a thrilling win.

Shubman Gill is a stylish young opener who has cemented his place in the Indian team. His backfoot punches and calm demeanor have drawn comparisons to Rahul Dravid and Virat Kohli. In 2023, Gill scored a double century against New Zealand, becoming the youngest Indian to do so in ODIs. He followed it up with centuries in Tests and T20s, proving his all-format capability.

KL Rahul is a technically sound batsman who also serves as a wicketkeeper. His versatility allows him to bat anywhere in the top or middle order. In the 2023 Asia Cup final, he returned from injury and scored a fluent 111*, guiding India to a title win. His ability to anchor innings and accelerate later makes him vital in ODIs and T20s.

Hardik Pandya is a hard-hitting all-rounder known for his explosive batting and useful fast bowling. He has won many close matches with his six-hitting ability, like the 2022 match against England where he scored 71 and took 4 wickets. As captain of Gujarat Titans, he led them to the IPL trophy in their debut season.

Mohammed Siraj emerged as a dependable pacer. His 6/21 spell in the Asia Cup 2023 final dismantled Sri Lanka, earning him the Player of the Match award. His journey from humble beginnings to becoming a key bowler in India's lineup is an inspiring story of perseverance.

Kuldeep Yadav, the left-arm wrist-spinner, brings variety and mystery. He has taken multiple five-wicket hauls and two international hat-tricks. In the 2023 World Cup, he was India's highest wicket-taker with his deceptive spin.

Rishabh Pant is known for his fearless cricket and unorthodox stroke play. His match-winning knock of 89* in Brisbane in 2021 helped India win the Test series in Australia against all odds. After recovering from a major accident, Pant returned stronger and continues to be a fan favorite.

Suryakumar Yadav, nicknamed SKY, has redefined T20 batting with his 360-degree strokeplay. His innings of 102 off 49 balls against South Africa in 2022 was hailed as one of the greatest T20 innings of all time. He is currently ranked among the top T20 batsmen globally.

India’s rich cricketing history includes epic wins such as the 1983 and 2011 ODI World Cup victories, the 2007 T20 World Cup triumph, and back-to-back Test series wins in Australia. The team has also won multiple Asia Cups and Champions Trophies.

India’s bench strength is a major asset. Players like Ishan Kishan, Sanju Samson, Ruturaj Gaikwad, Arshdeep Singh, Washington Sundar, and Prithvi Shaw are waiting for their chances and have proven their worth in IPL and India A matches.

Dravid, now the head coach, emphasizes discipline, technique, and long-term planning. Under his guidance, India focuses on grooming young talent, rotating players to manage workload, and maintaining high fitness levels.

The Indian cricket team also focuses heavily on fielding, fitness, and mental conditioning. Virat Kohli set the benchmark for fitness in Indian cricket, inspiring younger players to improve their strength and agility. Jadeja and Suryakumar are considered among the best fielders in the world.

Strategically, India uses data analysis, match simulations, and pitch-specific planning. The BCCI has invested in state-of-the-art training facilities and high-performance centers. Players work closely with batting and bowling coaches, mental conditioning experts, and physiotherapists.

India's rivalry with Pakistan continues to be the most-watched and intense clash in world cricket. Matches between the two nations often draw over a billion viewers. In the 2022 T20 World Cup match at the MCG, over 90,000 fans witnessed a last-over thriller that India won.

India is preparing for future ICC tournaments including the 2025 Champions Trophy, the 2026 T20 World Cup, and the 2027 ODI World Cup. The goal is to dominate all three formats and finally win an ICC title after over a decade.

The Indian men’s cricket team is more than just a sports team — it's a symbol of national pride, unity, and ambition. With a perfect blend of youth and experience, strategy and aggression, skill and spirit, India is ready to scale new heights in world cricket."""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([paragraph])

def next_word_predict(texts,num_word):
    print(texts)
    for i in range(num_word):
        token_text = tokenizer.texts_to_sequences([texts])[0]
        padded_token_text = pad_sequences([token_text], padding='pre', maxlen=77)
        pos = np.argmax(model.predict(padded_token_text))
        for word, index in tokenizer.word_index.items():
            if index == pos:
                texts = texts + " " + word
                print("answer = ", texts)
    return texts


st.title("Next Word Predictor")
st.markdown("Scrollable Paragraph Reference")
st.text_area("Reference Paragraph:", paragraph, height=170)

user_input = st.text_input("Enter starting text (e.g., virat kohli)", value="virat kohli")
num_words = st.slider("How many words to predict?", 1, 10, 5)

if st.button("Predict Next Words"):
    text=next_word_predict(user_input,num_words)
    st.success("Predicted Sentence: ")
    st.write(text)






# __author:Owner
# date:6/16/2022
import streamlit as st
import pandas as pd
from PIL import Image
#import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import os
import seaborn as sns
from wordcloud import WordCloud

#db_conn = mysql.connector.connect(**st.secrets["mysql"])
#cur = db_conn.cursor()

path = os.getcwd()

auto_word_list = ['battery', 'quality', 'tool', 'price', 'small', 'truck','plug', 'filter', 'oil', 'leather', 'air', 'water', 'clean', 'engine', 'bottle', 'kit','wax', 'paint', 'shine', 'clay', 'polish', 'products', 'wash', 'finish', 'spray', 'plastic', 'wheel', 		'tire','best', 'cover','cheap', 'hose', 'trailer', 'light', 'install','rv', 'fit']

mag_word_list = ['order', 'months', 'enjoy', 'computer', 'worth', 'reviews',  'reading', 'health', 'kindle', 'favorite', 'cover', 'pages', 'tips', 'issues', 'helpful', 'fashion', 'allure', 'handyman', 'home','informative','content', 'stuff','ads','gift', 'subscribed', 		'article']

gc_word_list = ['love', 'kindle' 'friend', 'birthday', 'christmas', 'email', 'money','wedding', 'thank', 'birthday', 'gifts', 'present','thanks', 'convenient','happy', 'printed']

sftw_word_list = ['microsoft', 'price', 'subscription', 'install', 'key', 'year', 'deal', 'computers', 'windows', 'card', 'ms', 'mac', 'spanish', 'japanese', 'language', 'learn', 'dvd', 'interface', 'computer', 'excel', 'support', 'problems', 'purchase', 'notes', 		'word']


def home_page_module():
    st.title("NLP - Amazon review Analysis")
    image = Image.open('nlp_homepage.jpeg')
    st.image(image, caption='** Amazon reviews analysis ** ')
    st.subheader("About Application:")
    st.write("The objective of our project is to analyze amazon reviews for certain categories like auto, magazine, gift cards and do the following:")
    st.write("1. Show Clean data for all categories to the user.")
    st.write("2. Topic modeling: Extract topics and see reviews pertaining to those. ")
    st.write("3. Sentiment Analysis: Perform sentiment analysis on reviews and predict the review sentiment.")
    st.write("4. Research relevant statistics about the reviews.")


def sentiment_analysis():
    pass


def plot_avg_by_state():
    viz_query_str = """ select avg(price) as average_price, state
                          from listing l
                          join city c on l.city_id = c.id
                         group by state
                         order by avg(price) desc;"""

    resultset = run_query(viz_query_str)
    resultdf = pd.DataFrame(resultset, columns=('Average Price', 'State'))

    fig, ax = plt.subplots()
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.barh(resultdf['State'], resultdf['Average Price'], align='center')
    ax.set_ylabel('States')
    ax.set_xlabel('Average Housing Price')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title('Average Housing Price by States')

    st.pyplot(fig)


def plot_avg_by_cities(inState):
    viz_query_str = """ select avg(price) as average_price, city
                          from listing l
                          join city c on l.city_id = c.id
                        where state = '""" + inState
    viz_query_str += """' group by city
                         order by avg(price) desc
                         limit 25;"""

    resultset = run_query(viz_query_str)
    resultdf = pd.DataFrame(resultset, columns=('Average Price', 'City'))

    fig, ax = plt.subplots()
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.barh(resultdf['City'], resultdf['Average Price'], align='center')
    ax.set_ylabel('Cities')
    ax.set_xlabel('Average Housing Price')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title('Average Housing Price by Cities')

    st.pyplot(fig)


def plot_avg_by_zip(inState):
    viz_query_str = """ select avg(price) as average_price, CONVERT(zip_code,char) as zip
                          from listing l
                          join city c on l.city_id = c.id
                        where state = '""" + inState
    viz_query_str += """' group by zip_code
                         order by avg(price) desc
                         limit 25;"""

    resultset = run_query(viz_query_str)
    resultdf = pd.DataFrame(resultset, columns=('Average Price', 'Zip'))

    fig, ax = plt.subplots()
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.barh(resultdf['Zip'], resultdf['Average Price'], align='center')
    ax.set_ylabel('Zip Code')
    ax.set_xlabel('Average Housing Price')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title('Average Housing Price by Zip Codes')

    st.pyplot(fig)


def main():
    if 'loggedIn' not in st.session_state:
        st.session_state.loggedIn = False

    menu = ["Home", "Search Reviews", "Topic Modeling",
            "Sentiment analysis", "Statistical Plots"]
    #menu = ["Home","Search Reviews" , "Topic Modeling","Sentiment analysis", "Statistical Plots", "Login", "SignUp" ]
    #menu = ["Home" ,"Topic Modeling","Sentiment analysis","Statistical Plots", "Login", "SignUp" ]

    choice = st.sidebar.selectbox("Menu", menu)

    category = ["Automotive", "Software",
                "Magazine Subscription", "Gift Cards"]
    option = st.sidebar.selectbox("Category", category)

    #option = st.selectbox( "Please Select Category" , ('-- Choose One -','Automotive','Software','Magazine Subscription','Gift Cards' ))
    if option == 'Automotive':
        auto_clean_reviews_url = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/auto_clean_reviews.csv?token=GHSAT0AAAAAAAAASIRZ4NL5ESAHPF4JH7UIY4VLCGA'
        auto_topics_url = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/auto_topics_ui.csv?token=GHSAT0AAAAAAAAASIRZRHCN6G3OUPCEYX3AY4VLDIA'
        auto_topic_reviews = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/auto_topic_review_ui.csv?token=GHSAT0AAAAAAAAASIRYOAQ4KTXSWKLZSQPSY4VLC4A'
        auto_sentiments = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/auto_sentiments.csv?token=GHSAT0AAAAAAAAASIRYRJMTIIIHCITETBZ2Y4VLCTQ'
        df_auto_clean_reviews = pd.read_csv(auto_clean_reviews_url)
        df_auto_topics = pd.read_csv(auto_topics_url)
        df_auto_topics_reviews = pd.read_csv(auto_topic_reviews)
        df_auto_sentiments = pd.read_csv(auto_sentiments)
        df_clean_reviews = df_auto_clean_reviews
        df_topic_review_ui = df_auto_topics_reviews
        df_topics_ui = df_auto_topics
        word_list = auto_word_list
        df_sentiments = df_auto_sentiments
        #df_word_cloud = df_auto_topic_prob
        im = 'auto_wordcloud.png'
    elif option == 'Software':
        sftw_clean_reviews_url = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/sftw_clean_reviews.csv?token=GHSAT0AAAAAAAAASIRYDIE7ECBVSZAD7LM6Y4VLN7A'
        sftw_topics_url = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/sftw_topics_ui.csv?token=GHSAT0AAAAAAAAASIRZITUI5URO7ZPHLOHCY4VLO4A'
        sftw_topic_reviews = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/sftw_topic_review_ui.csv?token=GHSAT0AAAAAAAAASIRYHZV5XPZR6P3DJA3QY4VLOUQ'
        sftw_sentiments = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/sftw_sentiments.csv?token=GHSAT0AAAAAAAAASIRY3SQ4P6JW6VVTT55QY4VLOLQ'
        df_sftw_clean_reviews = pd.read_csv(sftw_clean_reviews_url)
        df_sftw_topics = pd.read_csv(sftw_topics_url)
        df_sftw_topics_reviews = pd.read_csv(sftw_topic_reviews)
        df_sftw_sentiments = pd.read_csv(sftw_sentiments)
        df_clean_reviews = df_sftw_clean_reviews
        df_topic_review_ui = df_sftw_topics_reviews
        df_topics_ui = df_sftw_topics
        word_list = sftw_word_list
        df_sentiments = df_sftw_sentiments
        im = 'sftw_wordcloud.png'
    elif option == 'Gift Cards':
        gc_clean_reviews_url = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/gc_clean_reviews.csv?token=GHSAT0AAAAAAAAASIRYQMQYNICR4KOAT6CKY4VLLVA'
        gc_topics_url = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/gc_topics_ui.csv?token=GHSAT0AAAAAAAAASIRYT52KGMSIXLT5IU6QY4VLNIA'
        gc_topic_reviews = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/gc_topic_review_ui.csv?token=GHSAT0AAAAAAAAASIRZAL4RWUNRX764XGEAY4VLMOQ'
        gc_sentiments = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/gc_sentiments.csv?token=GHSAT0AAAAAAAAASIRZQ6YBM4WRPWHQOBTSY4VLMEA'
        df_gc_clean_reviews = pd.read_csv(gc_clean_reviews_url)
        df_gc_topics = pd.read_csv(gc_topics_url)
        df_gc_topics_reviews = pd.read_csv(gc_topic_reviews)
        df_gc_sentiments = pd.read_csv(gc_sentiments)
        df_clean_reviews = df_gc_clean_reviews
        df_topic_review_ui = df_gc_topics_reviews
        df_topics_ui = df_gc_topics
        word_list = gc_word_list
        df_sentiments = df_gc_sentiments
        im = 'gc_wordcloud.png'
    elif option == 'Magazine Subscription':
        mag_clean_reviews_url = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/mag_clean_reviews.csv?token=GHSAT0AAAAAAAAASIRYGMSZQI4JEGLDFUQSY4VLQ2A'
        mag_topics_url = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/mag_topics_ui.csv?token=GHSAT0AAAAAAAAASIRZFF2BCHLA6WGILLBMY4VLR2Q'
        mag_topic_reviews = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/mag_topic_review_ui.csv?token=GHSAT0AAAAAAAAASIRZ6RMTVNYWQA7OUV66Y4VLRMQ'
        mag_sentiments = 'https://raw.github.iu.edu/pyacham/NLP_FinalProject/main/mag_sentiments.csv?token=GHSAT0AAAAAAAAASIRZVWK3K67QZNOFKWFUY4VLRCQ'
        df_mag_clean_reviews = pd.read_csv(mag_clean_reviews_url)
        df_mag_topics = pd.read_csv(mag_topics_url)
        df_mag_topics_reviews = pd.read_csv(mag_topic_reviews)
        df_mag_sentiments = pd.read_csv(mag_sentiments)
        df_clean_reviews = df_mag_clean_reviews
        df_topic_review_ui = df_mag_topics_reviews
        df_topics_ui = df_mag_topics
        word_list = mag_word_list
        df_sentiments = df_mag_sentiments
        im = 'mag_wordcloud.png'

    df_sentiments['Actual_Sentiment'] = df_sentiments['overall'].apply(
        lambda overall: 'positive' if overall >= 3 else 'negative')


    # HOME PAGE
    if choice == "Home":
        home_page_module()

    # LOGIN PAGE AND SUBSEQUENT FUNCTIONS
    elif choice == "Login":
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            #hashed_pswd = make_hashes( password )
            # run_query('SELECT * FROM userstable WHERE username ="{}" and password="{}"'. format(username,check_hashes( password , hashed_pswd )))
            result = True

            if result:
                st.session_state.loggedIn = True
                st.success("successfully logged in as " + username)
            else:
                st.warning("incorrect usernmae/password")

    elif choice == "Sentiment analysis":
        st.title("Sentiment Analysis - " + option)
        with st.expander('Sentiment analysis'):

            df_sentiments = df_sentiments[['asin', 'reviewText', 'overall', 'Clean_Reviews', 'Actual_Sentiment',
                                           'textblob_polarity', 'textblob_sentiment', 'afin_polarity', 'afin_sentiment', 'vader_sentiment']]
            data = df_sentiments
            # (AgGrid(data))

            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_pagination(
                paginationAutoPageSize=True)  # Add pagination
            gb.configure_side_bar()  # Add a sidebar
            # Enable multi-row selection
            gb.configure_selection('multiple', use_checkbox=True,
                                   groupSelectsChildren="Group checkbox select children")
            gridOptions = gb.build()

            grid_response = AgGrid(
                data,
                gridOptions=gridOptions,
                data_return_mode='AS_INPUT',
                update_mode='MODEL_CHANGED',
                fit_columns_on_grid_load=False,
                theme="streamlit",
                # theme='ALPINE', #streamlit, balham,material
                enable_enterprise_modules=True,
                height=350,
                width='100%',
                reload_data=True
            )
            data = grid_response['data']
            selected = grid_response['selected_rows']
            df = pd.DataFrame(selected)

            # st.dataframe(df_sentiments.head())
        with st.expander('Count plot'):

            col_list = ['overall', 'Actual_Sentiment',
                        'textblob_sentiment', 'afin_sentiment', 'vader_sentiment']
            fig = plt.figure(figsize=(20, 20))
            for i, col in enumerate(col_list):
                ax = plt.subplot(3, 2, i+1)
                sns.countplot(data=df_sentiments[col_list], x=col, ax=ax,)
            plt.suptitle('Bar plot - Multiple predictors')
            sns.set(font_scale=2)
            st.pyplot(fig)
            #image = Image.open('nlp_homepage.jpeg')
            #st.image(image, caption='** Buy a Listing** ')

    elif choice == "Topic Modeling":
        #option = st.selectbox( "Please Select Category" , ('-- Choose One -','Automotive','Software','Magazine Subscription','Gift Cards' ))

        st.title("Topic Modelling - " + option)

        with st.expander('Topics'):
            st.dataframe(df_topics_ui.iloc[:, 1:].head())

        with st.expander('Topic Modeling'):
            word_select = ""
            word_select = st.selectbox("Please Select Word", (word_list))

            if len(word_select) > 1:
                derive_topic = df_topics_ui.apply(
                    lambda row: row[row == word_select], axis=1)

                derive_topic.columns[0]
                query = np.where(
                    df_topic_review_ui['predicted'] == derive_topic.columns[0])

                df_topic_review_ui = df_topic_review_ui[[
                    'asin', 'Clean_Reviews', 'predicted']]
                data = df_topic_review_ui.loc[query].head(10)
                gb = GridOptionsBuilder.from_dataframe(data)
                gb.configure_pagination(
                    paginationAutoPageSize=True)  # Add pagination
                gb.configure_side_bar()  # Add a sidebar
                # Enable multi-row selection
                gb.configure_selection(
                    'multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children")
                gridOptions = gb.build()

                grid_response = AgGrid(
                    data,
                    gridOptions=gridOptions,
                    data_return_mode='AS_INPUT',
                    update_mode='MODEL_CHANGED',
                    fit_columns_on_grid_load=False,
                    theme="streamlit",
                    # theme='ALPINE', #streamlit, balham,material
                    enable_enterprise_modules=True,
                    height=350,
                    width='100%',
                    reload_data=True
                )
                data = grid_response['data']
                selected = grid_response['selected_rows']
                df = pd.DataFrame(selected)
            else:
                st.info("Please select an appropriate word from Topic list")

        with st.expander('Topics Word Cloud'):

            image = Image.open(im)
            st.image(image, caption='** Word_Cloud ** ')

    # STATISTICAL VISUALIZATIONS
    elif choice == "Statistical Plots":
        st.title("Statistical Plots - " + option)
        # Code to calculate the word count from the dataframe.

        df_clean_reviews['word_count'] = df_clean_reviews['Clean_Reviews'].apply(
            lambda x: len(str(x).split(' ')))

        # Code to display the First Plot of Positive vs Negative Sentiments
        fig = plt.figure(figsize=(10, 4))
        plt.title("Positive vs Negative Sentiments")
        sns.countplot(data=df_clean_reviews, x='sentiment')
        st.pyplot(fig)

        # Code to display the Word Count vs Count of Reviews
        fig = plt.figure(figsize=(10, 4))
        plt.title("Word Count vs Count of Reviews")
        sns.countplot(data=df_clean_reviews, x=df_clean_reviews['word_count'])
        st.pyplot(fig)

        # df_clean_reviews['char_count'] = df_clean_reviews['Clean_Reviews'].apply(
        #     lambda x: len(x))

        # Code to display the overall rating split in the Dataframe.
        fig = plt.figure(figsize=(10, 4))
        plt.title("Overall Rating Split")
        sns.countplot(data=df_clean_reviews, x='overall')
        st.pyplot(fig)

######  END OF STATISTICAL PLOT PAGE ##################

    # SIGNUP PAGE
    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')

        if st.button("Signup"):
            #sql="INSERT INTO userstable(username,password) VALUES (%s,%s)"
            #val= (new_user , make_hashes(new_password))
            # cur.execute(sql,val)
            # cur.execute('commit')
            # add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")

    # SEARCH PAGE
    elif choice == "Search Reviews":
        st.title("Search Reviews - " + option)
        df_clean_reviews = df_clean_reviews[[
            'reviewerID', 'asin', 'reviewText', 'Clean_Reviews', 'overall', 'sentiment']]
        data = df_clean_reviews.head(200)
        # (AgGrid(data))

        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
        gb.configure_side_bar()  # Add a sidebar
        # Enable multi-row selection
        gb.configure_selection('multiple', use_checkbox=True,
                               groupSelectsChildren="Group checkbox select children")
        gridOptions = gb.build()

        grid_response = AgGrid(
            data,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT',
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=False,
            theme="streamlit",
            # theme='ALPINE', #streamlit, balham,material
            enable_enterprise_modules=True,
            height=350,
            width='100%',
            reload_data=True
        )
        data = grid_response['data']
        selected = grid_response['selected_rows']
        df = pd.DataFrame(selected)


if __name__ == '__main__':
    main()

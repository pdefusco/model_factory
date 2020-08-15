import sqlite3
conn = sqlite3.connect('newdb.db')

#df = pd.DataFrame({'best_score':1,'n_splits':2}, index=[0])

#df.to_sql(name='test_table', if_exists='append', con=conn)

#pd.read_sql('select * from experiments;', conn)

df.head()
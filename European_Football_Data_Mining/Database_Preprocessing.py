import sqlite3
import pandas as pd
import numpy

class SQLLiteConnection(object):

    def __init__(self, db_input):

        self.db_file = db_input
        self.conn = None


    def __enter__(self):
        print("Connecting to Database.")
        print(self.db_file)
        self.conn = sqlite3.connect(self.db_file)
        print("Connected.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database.")
        self.conn.close()

with SQLLiteConnection("E:\\Documents\\CSCI_6443\\database.sqlite") as lite:

    cursor = lite.conn.cursor()

    cursor.execute("SELECT league_id, match_api_id, home_team_api_id, away_team_api_id,"
                                   " home_team_goal, away_team_goal, home_player_X1, home_player_X2,"
                                   " home_player_X3, home_player_X4, home_player_X5, home_player_X6, "
                                   " home_player_X7, home_player_X8, home_player_X9 FROM Match")

    match_dataset = cursor.fetchall()
    print(match_dataset)

    cursor.execute("SELECT CASE WHEN home_team_goal>away_team_goal THEN 0 WHEN home_team_goal==away_team_goal THEN 1 ELSE 2 END FROM Match")
    result_labels = cursor.fetchall()

    print(match_dataset)


if __name__ == '__main__':
    #sup = SQLLiteConnection("E:\\Documents\\CSCI_6443\\database.sqlite")

    data_file = numpy.genfromtxt("E:\\Documents\\CSCI_6443\\Data_Mining_Project\\final_dataset.csv", skip_header=1, delimiter=',', usecols=range(0, 32))
    print(data_file)

    numpy.random.shuffle(data_file)

    numpy.savetxt("randomized_final_dataset.csv", data_file, fmt="%s", delimiter=",")


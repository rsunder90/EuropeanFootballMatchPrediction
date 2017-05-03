/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package soccer.preprocessing;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import soccer.dao.Match_DAO;
import soccer.dao.NewTable;
import soccer.database.SQLiteJDBCDriverConnection;
import soccer.utilities.Utilities;

/**
 *
 * @author manuel
 */


public class SQLitePreprocessing {
    
    public static ArrayList<Match_DAO> selectMatches(){
        
        String sql = "SELECT * "
                    +"FROM Match";
        
        ArrayList<Match_DAO> Matches = new ArrayList();
        
        try (Connection conn = SQLiteJDBCDriverConnection.connect();
             Statement stmt  = conn.createStatement();
             ResultSet rs    = stmt.executeQuery(sql)){
            
            while(rs.next()){
                Match_DAO match = new Match_DAO();
                
                match.setId(rs.getInt("id"));
                match.setLeague_id(rs.getInt("league_id"));
                match.setSeason(rs.getString("season"));
                match.setDate(rs.getString("date"));
                match.setHome_team_id(rs.getInt("home_team_api_id"));
                match.setAway_team_id(rs.getInt("away_team_api_id"));
                match.setHome_team_goals(rs.getInt("home_team_goal"));
                match.setAway_team_goals(rs.getInt("away_team_goal"));
                Matches.add(match);
            }
            
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
        return Matches;
    }
    
    public static void main(String[] args) {
        ArrayList<Match_DAO> matches = selectMatches();
        HashMap<Integer, ArrayList<Match_DAO>> matchesPerTeam = Utilities.createCompressingHashMap(matches);
        ArrayList<NewTable> newTable = new ArrayList<>();
        Integer a = 0;

       
        for(Match_DAO match : matches){
            NewTable n = new NewTable();
            n.setMatch_id(match.getId());
            
            Integer home_team_goals = Utilities.findHomeGoalSoFarInTheSeason(match.getHome_team_id(), match.getSeason(), match.getDate(), matchesPerTeam);
            Integer home_teams_goals_conceded = Utilities.findHomeConcededGoalSoFarInTheSeason(match.getHome_team_id(), match.getSeason(), match.getDate(), matchesPerTeam);
            Integer home_games = Utilities.findHomeMatchesSoFarInTheSeason(match.getHome_team_id(), match.getSeason(), match.getDate(), matchesPerTeam);
            
            Integer all_home_goals = Utilities.findAllHomeGoalSoFarInTheSeason(match.getLeagueId(), match.getSeason(), match.getDate(), matches);
            Integer all_home_conceded_goals = Utilities.findAllHomeConcededGoalSoFarInTheSeason(match.getLeagueId(), match.getSeason(), match.getDate(), matches);
            Integer all_games = Utilities.findAllGamesSoFarInTheSeason(match.getLeagueId(), match.getSeason(), match.getDate(), matches);
            
            Float home_team_attack = home_games.equals(0) ? 0 : (float)home_team_goals/(float)home_games;
            Float all_home_team_attack = all_games.equals(0) ? 0 : (float) all_home_goals/ (float) all_games;
            Float home_team_relative_attack = all_home_team_attack.equals((float)0) ? 0 : home_team_attack / all_home_team_attack; 
            
            Float home_team_deffense = home_games.equals(0) ? 0 : (float)home_teams_goals_conceded/(float)home_games;
            Float all_home_team_deffense = all_games.equals(0) ? 0 : (float) all_home_conceded_goals/ (float) all_games;
            Float home_team_relative_deffense = all_home_team_deffense.equals((float)0) ? 0 : home_team_deffense / all_home_team_deffense;
            
            Integer away_team_goals = Utilities.findAwayGoalSoFarInTheSeason(match.getHome_team_id(), match.getSeason(), match.getDate(), matchesPerTeam);
            Integer away_teams_goals_conceded = Utilities.findAwayConcededGoalSoFarInTheSeason(match.getHome_team_id(), match.getSeason(), match.getDate(), matchesPerTeam);
            Integer away_games = Utilities.findAwayMatchesSoFarInTheSeason(match.getHome_team_id(), match.getSeason(), match.getDate(), matchesPerTeam);
            
            Float away_team_attack = away_games.equals(0) ? 0 : (float)away_team_goals/(float)away_games;
            Float all_away_team_attack = all_games.equals(0) ? 0 : (float) all_home_conceded_goals/ (float) all_games;
            Float away_team_relative_attack = all_away_team_attack.equals((float)0) ? 0 : away_team_attack / all_away_team_attack; 
            
            Float away_team_deffense = away_games.equals(0) ? 0 : (float)away_teams_goals_conceded/(float)away_games;
            Float all_away_team_deffense = all_games.equals(0) ? 0 : (float) all_home_goals/ (float) all_games;
            Float away_team_relative_deffense = all_away_team_deffense.equals((float)0) ? 0 : away_team_deffense / all_away_team_deffense;
            
            n.setHome_attack(home_team_relative_attack);
            n.setHome_defense(home_team_relative_deffense);
            
            n.setAway_attack(away_team_relative_attack);
            n.setAway_deffense(away_team_relative_deffense);
            newTable.add(n);
        }
        
        //Export to CSV
        
        writeFile(newTable);
        
    }
    
    private static void writeFile(ArrayList<NewTable> n){
        
        String FILENAME = "calculations.csv";

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(FILENAME))) {

                String header = "match_id, home_attack, home_deffense, away_attack, away_deffense\n";

                bw.write(header);
                for(NewTable match : n){
                    bw.write(match.toString());
                }
                System.out.println("Done");

        } catch (IOException e) {

                e.printStackTrace();

        }
        
    }
    
}

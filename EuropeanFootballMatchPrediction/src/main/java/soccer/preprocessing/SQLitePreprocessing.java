/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package soccer.preprocessing;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import soccer.database.SQLiteJDBCDriverConnection;

/**
 *
 * @author manuel
 */


public class SQLitePreprocessing {
    
    public static void basicSelect(){
        
        String sql = "SELECT Match.id, home_team_api_id, away_team_api_id, \n" +
                            "a.buildUpPlaySpeed, a.buildUpPlaySpeedClass, a.buildUpPlayDribblingClass, a.buildUpPlayPassing, "
                          + "a.buildUpPlayPassingClass, a.buildUpPlayPositioningClass, a.chanceCreationPassing, "
                          + "a.chanceCreationPassingClass, a.chanceCreationCrossing, a.chanceCreationCrossingClass, "
                          + "a.chanceCreationShooting, a.chanceCreationShootingClass, a.chanceCreationPositioningClass, "
                          + "a.defencePressure, a.defencePressureClass, a.defenceAggression, a.defenceAggressionClass, "
                          + "a.defenceTeamWidth, a.defenceTeamWidthClass, a.defenceDefenderLineClass,\n" +
                            "b.buildUpPlaySpeed, b.buildUpPlaySpeedClass, b.buildUpPlayDribblingClass, b.buildUpPlayPassing, "
                          + "b.buildUpPlayPassingClass, b.buildUpPlayPositioningClass, b.chanceCreationPassing, "
                          + "b.chanceCreationPassingClass, b.chanceCreationCrossing, b.chanceCreationCrossingClass, "
                          + "b.chanceCreationShooting, b.chanceCreationShootingClass, b.chanceCreationPositioningClass, "
                          + "b.defencePressure, b.defencePressureClass, b.defenceAggression, b.defenceAggressionClass, "
                          + "b.defenceTeamWidth, b.defenceTeamWidthClass, b.defenceDefenderLineClass\n" +
                    "FROM (Match LEFT JOIN Team_Attributes a ON (Match.home_team_api_id = a.team_api_id AND substr(Match.season,1,4) = strftime('%Y', a.date)))\n" +
                          "LEFT JOIN Team_Attributes b ON (Match.away_team_api_id = b.team_api_id AND substr(Match.season,1,4) = strftime('%Y', b.date))";
        
        try (Connection conn = SQLiteJDBCDriverConnection.connect();
             Statement stmt  = conn.createStatement();
             ResultSet rs    = stmt.executeQuery(sql)){
            
            System.out.println(rs.getInt(1));
            
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        basicSelect();
    }
    
}

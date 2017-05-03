/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package soccer.utilities;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import soccer.dao.Match_DAO;

/**
 *
 * @author manuel
 */
public class Utilities {
    
    public static Integer findHomeGoalSoFarInTheSeason(Integer team_id, String season, String matchDate, HashMap<Integer, ArrayList<Match_DAO>> matches_per_team){
        
        ArrayList<Match_DAO> teamMatches = matches_per_team.get(team_id);
        Integer goals = 0;
        for(Match_DAO match : teamMatches){
            if(match.getHome_team_id().equals(team_id) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                goals += match.getHome_team_goals();
            }
        }
        
        return goals;
        
    }
    
    public static Integer findAllHomeGoalSoFarInTheSeason(Integer ligueId, String season, String matchDate, ArrayList<Match_DAO> matches){
        
        Integer goals = 0;
        for(Match_DAO match : matches){
            if(ligueId.equals(match.getLeagueId()) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                goals += match.getHome_team_goals();
            }
        }
        
        return goals;
        
    }
    
    public static Integer findAllHomeConcededGoalSoFarInTheSeason(Integer ligueId, String season, String matchDate, ArrayList<Match_DAO> matches){
        
        Integer goals = 0;
        for(Match_DAO match : matches){
            if(ligueId.equals(match.getLeagueId()) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                goals += match.getAway_team_goals();
            }
        }
        
        return goals;
        
    }
    
    public static Integer findAllGamesSoFarInTheSeason(Integer ligueId, String season, String matchDate, ArrayList<Match_DAO> matches){
        
        Integer games = 0;
        for(Match_DAO match : matches){
            if(ligueId.equals(match.getLeagueId()) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                games++;
            }
        }
        
        return games;
        
    }
    
    public static Integer findHomeConcededGoalSoFarInTheSeason(Integer team_id, String season, String matchDate, HashMap<Integer, ArrayList<Match_DAO>> matches_per_team){
        
        ArrayList<Match_DAO> teamMatches = matches_per_team.get(team_id);
        Integer goals = 0;
        for(Match_DAO match : teamMatches){
            if(match.getHome_team_id().equals(team_id) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                goals += match.getAway_team_goals();
            }
        }
        
        return goals;
        
    }
    
    public static Integer findHomeMatchesSoFarInTheSeason(Integer team_id, String season, String matchDate, HashMap<Integer, ArrayList<Match_DAO>> matches_per_team){
        
        ArrayList<Match_DAO> teamMatches = matches_per_team.get(team_id);
        Integer matches = 0;
        for(Match_DAO match : teamMatches){
            if(match.getHome_team_id().equals(team_id) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                matches++;
            }
        }
        
        return matches;
    }
    
    
    public static Integer findAwayGoalSoFarInTheSeason(Integer team_id, String season, String matchDate, HashMap<Integer, ArrayList<Match_DAO>> matches_per_team){
        
        ArrayList<Match_DAO> teamMatches = matches_per_team.get(team_id);
        Integer goals = 0;
        for(Match_DAO match : teamMatches){
            if(match.getAway_team_id().equals(team_id) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                goals += match.getAway_team_goals();
            }
        }
        
        return goals;
        
    }
    
    public static Integer findAwayConcededGoalSoFarInTheSeason(Integer team_id, String season, String matchDate, HashMap<Integer, ArrayList<Match_DAO>> matches_per_team){
        
        ArrayList<Match_DAO> teamMatches = matches_per_team.get(team_id);
        Integer goals = 0;
        for(Match_DAO match : teamMatches){
            if(match.getAway_team_id().equals(team_id) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                goals += match.getHome_team_goals();
            }
        }
        
        return goals;
        
    }
    
    public static Integer findAwayMatchesSoFarInTheSeason(Integer team_id, String season, String matchDate, HashMap<Integer, ArrayList<Match_DAO>> matches_per_team){
        
        ArrayList<Match_DAO> teamMatches = matches_per_team.get(team_id);
        Integer matches = 0;
        for(Match_DAO match : teamMatches){
            if(match.getAway_team_id().equals(team_id) && match.getSeason().equals(season) && isPreviousSeasonMatch(matchDate, match.getDate(), season)){
                matches++;
            }
        }
        
        return matches;
    }
    
    public static boolean isPreviousSeasonMatch(String matchDate, String otherMatchPlayed, String season){
        Date baseMatch = null;
        Date otherMatch = null;
        try {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
            baseMatch = sdf.parse(matchDate);
            otherMatch = sdf.parse(otherMatchPlayed);
            
            } catch (ParseException ex) {
                Logger.getLogger(Utilities.class.getName()).log(Level.SEVERE, null, ex);
        }
        return (otherMatch.before(baseMatch) && otherMatch.after(seasonInitialDate(season)));
    }
    
    public static HashMap<Integer, ArrayList<Match_DAO>> createCompressingHashMap(ArrayList<Match_DAO> matches){
        HashMap<Integer, ArrayList<Match_DAO>> compressingMap = new HashMap<>();
        for(Match_DAO match : matches){
            
            if(!compressingMap.containsKey(match.getHome_team_id())){
                
                ArrayList<Match_DAO> newList = new ArrayList<>();
                newList.add(match);
                compressingMap.put(match.getHome_team_id(), newList);
                
            } else {
                
                compressingMap.get(match.getHome_team_id()).add(match);
                
            }
            
            if(!compressingMap.containsKey(match.getAway_team_id())){
                
                ArrayList<Match_DAO> newList = new ArrayList<>();
                newList.add(match);
                compressingMap.put(match.getAway_team_id(), newList);
                
            } else {
                
                compressingMap.get(match.getAway_team_id()).add(match);
                
            }
            
        }
        return compressingMap;
    }
    
    public static Date seasonInitialDate(String season){
        Integer year = Integer.parseInt(season.substring(0, 4));
        Date date = new GregorianCalendar(year, 6, 30).getTime();
        return date;
    }
    
    public static Date seasonFinalDate(String season){
        Integer year = Integer.parseInt(season.substring(5, 9));
        Date date = new GregorianCalendar(year, 6, 30).getTime();
        return date;
    }
    
}

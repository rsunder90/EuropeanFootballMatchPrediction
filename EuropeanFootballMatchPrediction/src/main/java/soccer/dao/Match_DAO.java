/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package soccer.dao;

/**
 *
 * @author manuel
 */
public class Match_DAO {
    
    Integer id;
    Integer home_team_id;
    Integer away_team_id;
    Integer home_team_goals;
    Integer away_team_goals;
    String season;
    String date;
    Integer leagueId;

    public Match_DAO() {
    }

    public Match_DAO(Integer id, Integer home_team_id, Integer away_team_id, Integer home_team_goals, Integer away_team_goals, String season, String date, Integer leagueId) {
        this.id = id;
        this.home_team_id = home_team_id;
        this.away_team_id = away_team_id;
        this.home_team_goals = home_team_goals;
        this.away_team_goals = away_team_goals;
        this.season = season;
        this.date = date;
        this.leagueId = leagueId;
    }
    
    

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public Integer getHome_team_id() {
        return home_team_id;
    }

    public void setHome_team_id(Integer home_team_id) {
        this.home_team_id = home_team_id;
    }

    public Integer getAway_team_id() {
        return away_team_id;
    }

    public void setAway_team_id(Integer away_team_id) {
        this.away_team_id = away_team_id;
    }

    public Integer getHome_team_goals() {
        return home_team_goals;
    }

    public void setHome_team_goals(Integer home_team_goals) {
        this.home_team_goals = home_team_goals;
    }

    public Integer getAway_team_goals() {
        return away_team_goals;
    }

    public void setAway_team_goals(Integer away_team_goals) {
        this.away_team_goals = away_team_goals;
    }

    public String getSeason() {
        return season;
    }

    public void setSeason(String season) {
        this.season = season;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public Integer getLeagueId() {
        return leagueId;
    }

    public void setLeague_id(Integer leagueId) {
        this.leagueId = leagueId;
    }
    
}

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
public class NewTable {
    
    Integer match_id;
    Float home_attack;
    Float home_defense;
    Float away_attack;
    Float away_deffense;

    public NewTable() {
    }

    public NewTable(Integer match_id, Float home_attack, Float home_defense, Float away_attack, Float away_deffense) {
        this.match_id = match_id;
        this.home_attack = home_attack;
        this.home_defense = home_defense;
        this.away_attack = away_attack;
        this.away_deffense = away_deffense;
    }

    public Integer getMatch_id() {
        return match_id;
    }

    public void setMatch_id(Integer match_id) {
        this.match_id = match_id;
    }

    public Float getHome_attack() {
        return home_attack;
    }

    public void setHome_attack(Float home_attack) {
        this.home_attack = home_attack;
    }

    public Float getHome_defense() {
        return home_defense;
    }

    public void setHome_defense(Float home_defense) {
        this.home_defense = home_defense;
    }

    public Float getAway_attack() {
        return away_attack;
    }

    public void setAway_attack(Float away_attack) {
        this.away_attack = away_attack;
    }

    public Float getAway_deffense() {
        return away_deffense;
    }

    public void setAway_deffense(Float away_deffense) {
        this.away_deffense = away_deffense;
    }

    @Override
    public String toString() {
        return match_id + ", " + home_attack + ", " + home_defense + ", " + away_attack + ", " + away_deffense + "\n";
    }
    
}

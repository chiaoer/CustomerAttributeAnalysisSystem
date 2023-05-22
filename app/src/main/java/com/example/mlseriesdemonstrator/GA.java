package com.example.mlseriesdemonstrator;

import java.io.Serializable;

public class GA implements Serializable {

    public int gender;
    public int age;
    public enum videoType {
        Child,
        Young,
        MaleAdult,
        FemaleAdult,
        MaleSenior,
        FemaleSenior
    }
    public videoType videoType;
}

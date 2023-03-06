package com.example.mlseriesdemonstrator;

import com.google.mlkit.vision.face.Face;

public class FaceExtension {

    /******** Original Face Class parameters ********/
    public Face faceOri;

    /********* New Class parameters ***********/
    public int count = 0;//for face attribute inference
    public GA ga_result = null;
   // boolean isExisted = false;//whether this face is existed in the last faceList
    public int valid_count = 0;//Determine whether it is a valid face

    public long startTime = 0;//For calculating keep time.
    public long keepTime = 0;

    /********* New Class functions ***********/

//    public void setIsExisted(boolean isExisted) {
//        this.isExisted = isExisted;
//    }
//
//    public boolean getIsExisted() {
//        return this.isExisted;
//    }
}

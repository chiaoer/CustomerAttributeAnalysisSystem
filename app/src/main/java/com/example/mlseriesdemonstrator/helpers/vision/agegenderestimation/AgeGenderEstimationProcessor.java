package com.example.mlseriesdemonstrator.helpers.vision.agegenderestimation;

import static com.example.mlseriesdemonstrator.GA.videoType;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageProxy;

import com.example.mlseriesdemonstrator.FaceExtension;
import com.example.mlseriesdemonstrator.GA;
import com.example.mlseriesdemonstrator.helpers.vision.FaceGraphic;
import com.example.mlseriesdemonstrator.helpers.vision.GraphicOverlay;
import com.example.mlseriesdemonstrator.helpers.vision.VisionBaseProcessor;
import com.example.mlseriesdemonstrator.object.VisitorAnalysisActivity;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class AgeGenderEstimationProcessor extends VisionBaseProcessor<List<Face>> {

    public interface AgeGenderCallback {
        void onFaceDetected(FaceExtension face, int age, int gender, boolean keep);
    }

    private static final String TAG = "AgeGenderEstimationProcessor";

    // Input image size for our age model
    private static final int AGE_INPUT_IMAGE_SIZE = 200;

    // Input image size for our gender model
    private static final int GENDER_INPUT_IMAGE_SIZE = 128;

    private final FaceDetector detector;
    private final Interpreter ageModelInterpreter;
    private final ImageProcessor ageImageProcessor;
    private final Interpreter genderModelInterpreter;
    private final ImageProcessor genderImageProcessor;
    private final GraphicOverlay graphicOverlay;
    private final AgeGenderCallback callback;

    public VisitorAnalysisActivity activity;

//    HashMap<Integer, Integer> faceIdAgeMap = new HashMap<>();
//    HashMap<Integer, Integer> faceIdGenderMap = new HashMap<>();

    private ArrayList<FaceExtension> mTrackedFace;

    public AgeGenderEstimationProcessor(Interpreter ageModelInterpreter,
                                        Interpreter genderModelInterpreter,
                                        GraphicOverlay graphicOverlay,
                                        AgeGenderCallback callback) {
        this.callback = callback;
        this.graphicOverlay = graphicOverlay;
        // initialize processors
        this.ageModelInterpreter = ageModelInterpreter;
        ageImageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeOp(AGE_INPUT_IMAGE_SIZE, AGE_INPUT_IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0f, 255f))
                        .build();

        this.genderModelInterpreter = genderModelInterpreter;
        genderImageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(GENDER_INPUT_IMAGE_SIZE, GENDER_INPUT_IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0f, 255f))
                .build();

        FaceDetectorOptions faceDetectorOptions = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                // to ensure we don't count and analyse same person again
                .enableTracking()
                .build();
        detector = FaceDetection.getClient(faceDetectorOptions);
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    public Task<List<Face>> detectInImage(ImageProxy imageProxy, Bitmap bitmap, int rotationDegrees) {
        InputImage inputImage = InputImage.fromMediaImage(imageProxy.getImage(), rotationDegrees);
        int rotation = rotationDegrees;

        // In order to correctly display the face bounds, the orientation of the analyzed
        // image and that of the viewfinder have to match. Which is why the dimensions of
        // the analyzed image are reversed if its rotation information is 90 or 270.
        boolean reverseDimens = rotation == 90 || rotation == 270;
        int width;
        int height;
        if (reverseDimens) {
            width = imageProxy.getHeight();
            height =  imageProxy.getWidth();
        } else {
            width = imageProxy.getWidth();
            height = imageProxy.getHeight();
        }
        return detector.process(inputImage)
            .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                @Override
                public void onSuccess(List<Face> faces) {
                    graphicOverlay.clear();
                    if (faces.size() > 0)
                        Log.d(TAG, "## faces size: " + faces.size());

                    ArrayList<FaceExtension> newFaceList = new ArrayList<>();

                    for (Face face : faces) {
                        Log.d(TAG, "face found, id: " + face.getTrackingId());

                        //Copy the detected face to faceExtItem
                        FaceExtension faceExtItem = new FaceExtension();
                        faceExtItem.faceOri = face;

                        int trackId = face.getTrackingId();
                        boolean is_found = false;

                        if (mTrackedFace != null) {
                            Log.d(TAG, "mTrackedFace...size : " + mTrackedFace.size());

                            for (FaceExtension faceExt : mTrackedFace) {
                                if (faceExt.faceOri.getTrackingId() == trackId) {
                                    // faceExt.setIsExisted(true);
                                    faceExtItem.valid_count = faceExt.valid_count;
                                    faceExtItem.startTime = faceExt.startTime;
                                    faceExtItem.keepForVideo = faceExt.keepForVideo;

                                    faceExtItem.ga_result = faceExt.ga_result;
                                    faceExtItem.count = faceExt.count + 1;
                                    if (faceExtItem.count == 2) {
                                        faceExtItem.valid_count = faceExt.valid_count + 1;
                                        faceExtItem.count = 0; //reset
                                    }
                                    Log.d(TAG, "Face Keep Stay...count = " + faceExtItem.count + ", id = " + faceExtItem.faceOri.getTrackingId() + ", valid_count = " + faceExtItem.valid_count);

                                    if (faceExtItem.valid_count >= 1) {
                                        Log.d(TAG, "# valid_count >=1");

                                        //Show graphicOverlay
                                        FaceGraphic faceGraphic = new FaceGraphic(graphicOverlay, face, false, width, height);
                                        faceGraphic.age = (int) faceExtItem.ga_result.age;
                                        faceGraphic.gender = faceExtItem.ga_result.gender;
                                        graphicOverlay.add(faceGraphic);

                                        //Calculate
                                        long timeDiff = System.currentTimeMillis() - faceExtItem.startTime;
                                        Log.d(TAG, "# timeDiff = " + timeDiff + ", keepForVideo = " + faceExtItem.keepForVideo);

                                        if (timeDiff > 5000 && faceExtItem.keepForVideo == 0) {
                                            faceExtItem.keepForVideo = 1;

                                            if (callback != null) {
                                                callback.onFaceDetected(faceExtItem, (int) faceExtItem.ga_result.age, faceExtItem.ga_result.gender, true);
                                            }
                                        } else {
                                            if (callback != null) {
                                                callback.onFaceDetected(faceExtItem, (int) faceExtItem.ga_result.age, faceExtItem.ga_result.gender, false);
                                            }
                                        }
                                    }

                                    newFaceList.add(faceExtItem);
                                    is_found = true;
                                    break;
                                }
                            }
                        }

                        // New Face
                        if (!is_found) {
                            faceExtItem.startTime = System.currentTimeMillis();
                            Log.d(TAG, "# New Face startTime = " + faceExtItem.startTime);

                            // now we have a face, so we can use that to analyse age and gender
                            Bitmap faceBitmap = cropToBBox(bitmap, face.getBoundingBox(), rotation);

                            if (faceBitmap == null) {
                                Log.d("GraphicOverlay", "Face bitmap null");
                                return;
                            }

                            TensorImage tensorImage = TensorImage.fromBitmap(faceBitmap);
                            ByteBuffer ageImageByteBuffer = ageImageProcessor.process(tensorImage).getBuffer();
                            float[][] ageOutputArray = new float[1][1];
                            ageModelInterpreter.run(ageImageByteBuffer, ageOutputArray);

                            // The model returns a normalized value for the age i.e in range ( 0 , 1 ].
                            // To get the age, we multiply the model's output with p.
                            float age = ageOutputArray[0][0] * 116;
                            Log.d(TAG, "face id: " + face.getTrackingId() + ", age: " + age);

                            ByteBuffer genderImageByteBuffer = genderImageProcessor.process(tensorImage).getBuffer();
                            float[][] genderOutputArray = new float[1][2];
                            genderModelInterpreter.run(genderImageByteBuffer, genderOutputArray);
                            int gender;
                            if (genderOutputArray[0][0] > genderOutputArray[0][1]) {
                                // "Male"
                                gender = 1;
                            } else {
                                // "Female"
                                gender = 0;
                            }
                            Log.d(TAG, "face id: " + face.getTrackingId() + ", gender: " + (gender == 1 ? "Male" : "Female"));

                            faceBitmap.recycle();

                            faceExtItem.ga_result = new GA();
                            faceExtItem.ga_result.age = age;
                            faceExtItem.ga_result.gender = gender;

                            if (age < 13) {
                                faceExtItem.ga_result.videoType = videoType.Child;
                            } else if (age < 31) {
                                faceExtItem.ga_result.videoType = videoType.Young;
                            } else if (age < 61) {
                                if (gender == 1)
                                    faceExtItem.ga_result.videoType = videoType.MaleAdult;
                                else
                                    faceExtItem.ga_result.videoType = videoType.FemaleAdult;
                            } else {
                                if (gender == 1)
                                    faceExtItem.ga_result.videoType = videoType.MaleSenior;
                                else
                                    faceExtItem.ga_result.videoType = videoType.FemaleSenior;
                            }
                            newFaceList.add(faceExtItem);
                        }
                    }

                    mTrackedFace = newFaceList;
                }
            })
            .addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {
                    // intentionally left empty
                }
            });
    }

    public void stop() {
        detector.close();
    }

    private Bitmap cropToBBox(Bitmap image, Rect boundingBox, int rotation) {
        int shift = 0;
        if (rotation != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotation);
            image = Bitmap.createBitmap(image, 0, 0, image.getWidth(), image.getHeight(), matrix, true);
        }
        if (boundingBox.top >= 0 && boundingBox.bottom <= image.getWidth()
                && boundingBox.top + boundingBox.height() <= image.getHeight()
                && boundingBox.left >= 0
                && boundingBox.left + boundingBox.width() <= image.getWidth()) {
            return Bitmap.createBitmap(
                    image,
                    boundingBox.left,
                    boundingBox.top + shift,
                    boundingBox.width(),
                    boundingBox.height()
            );
        } else return null;
    }
}

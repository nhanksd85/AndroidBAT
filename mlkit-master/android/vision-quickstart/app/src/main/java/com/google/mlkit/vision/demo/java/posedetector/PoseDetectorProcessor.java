/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.posedetector;

import android.content.Context;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;

import androidx.annotation.NonNull;
import com.google.android.gms.tasks.Task;
import com.google.android.odml.image.MlImage;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.CameraXLivePreviewActivity;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.demo.java.posedetector.classification.PoseClassifierProcessor;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase;
import com.google.mlkit.vision.pose.PoseLandmark;

import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/** A processor to run pose detector. */
public class PoseDetectorProcessor
    extends VisionProcessorBase<PoseDetectorProcessor.PoseWithClassification> {
  private static final String TAG = "PoseDetectorProcessor";
  public boolean startEvaluation = false;
  private final PoseDetector detector;

  private final boolean showInFrameLikelihood;
  private final boolean visualizeZ;
  private final boolean rescaleZForVisualization;
  private final boolean runClassification;
  private final boolean isStreamMode;
  private final Context context;
  private final Executor classificationExecutor;
  public ImageButton startButton, endButton;
  private PoseClassifierProcessor poseClassifierProcessor;
  /** Internal class to hold Pose and classification results. */
  protected static class PoseWithClassification {
    private final Pose pose;
    private final List<String> classificationResult;

    public PoseWithClassification(Pose pose, List<String> classificationResult) {
      this.pose = pose;
      this.classificationResult = classificationResult;
    }

    public Pose getPose() {
      return pose;
    }

    public List<String> getClassificationResult() {
      return classificationResult;
    }
  }

  public PoseDetectorProcessor(
      Context context,
      PoseDetectorOptionsBase options,
      boolean showInFrameLikelihood,
      boolean visualizeZ,
      boolean rescaleZForVisualization,
      boolean runClassification,
      boolean isStreamMode) {
    super(context);
    this.showInFrameLikelihood = showInFrameLikelihood;
    this.visualizeZ = visualizeZ;
    this.rescaleZForVisualization = rescaleZForVisualization;
    detector = PoseDetection.getClient(options);
    this.runClassification = runClassification;
    this.isStreamMode = isStreamMode;
    this.context = context;
    classificationExecutor = Executors.newSingleThreadExecutor();
    Timer aTimer = new Timer();
    TimerTask aTask = new TimerTask() {
      @Override
      public void run() {
        DFA_Run();
        timer_run();
      }
    };
    aTimer.schedule(aTask, 1000, 1000);
  }


  int[] timer_counter = new int[10];
  int[] timer_flag = new int[10];
  private void setTimer(int index, int duration){
    timer_flag[index] = 0;
    timer_counter[index] = duration;
  }
  private void timer_run(){
    for(int i = 0; i < 10; i++){
      if(timer_counter[i] > 0){
        timer_counter[i] --;
        if(timer_counter[i] <=0) timer_flag[i] = 1;
      }
    }
  }
  List<PoseLandmark> landmarks;
  int status = IDLE;
  boolean isBodyDetection = false;
  private static final int IDLE = 0;
  private static final int START_EVAL = 1;
  private static final int END_EVAL = 2;
  List<Double> listLeftHipAngle = new ArrayList<Double>();
  List<Double> listRightHipAngle = new ArrayList<Double>();
  List<PoseLandmark> listLeftShouderPoint = new ArrayList<PoseLandmark>();
  List<PoseLandmark> listRightShouderPoint = new ArrayList<PoseLandmark>();

  List<PoseLandmark> listSigleLeftFoot = new ArrayList<PoseLandmark>();
  List<PoseLandmark> listSigleRightFoot = new ArrayList<PoseLandmark>();

  private void DFA_Run(){
    switch (status){
      case IDLE:
        if(startEvaluation == true && isBodyDetection == true){
          setTimer(0, 5);
          setTimer(1, 1);
          status = START_EVAL;
          Log.d("BAT", "Start BAT evaluation");
          listLeftHipAngle.clear();
          listRightHipAngle.clear();
          listLeftShouderPoint.clear();
          listRightShouderPoint.clear();
          listSigleLeftFoot.clear();
          listSigleRightFoot.clear();
        }
        break;
      case START_EVAL:
        if(timer_flag[1] == 1){
          setTimer(1, 1);
          listLeftHipAngle.add(angle_left);
          listRightHipAngle.add(angle_right);
          listLeftShouderPoint.add(leftShouder);
          listRightShouderPoint.add(rightShouder);

          listSigleLeftFoot.add(leftFoot);
          listSigleRightFoot.add(rightFoot);
        }
        if(timer_flag[0] == 1){
          status = END_EVAL;
          Log.d("BAT", "Left Hip: " + getDeveriationOfList(listLeftHipAngle));
          Log.d("BAT", "Right Hip: " + getDeveriationOfList(listRightHipAngle));

          double sumShouder = 0;
          for(int i = 0; i < listLeftShouderPoint.size() - 1; i++){
            double vector1_x = listLeftShouderPoint.get(i).getPosition().x - listRightShouderPoint.get(i).getPosition().x;
            double vector1_y = listLeftShouderPoint.get(i).getPosition().y - listRightShouderPoint.get(i).getPosition().y;

            double vector2_x = listLeftShouderPoint.get(i+1).getPosition().x - listRightShouderPoint.get(i+1).getPosition().x;
            double vector2_y = listLeftShouderPoint.get(i+1).getPosition().y - listRightShouderPoint.get(i+1).getPosition().y;

            double distance = Math.sqrt ((vector1_x - vector2_x) * (vector1_x - vector2_x)
                     + (vector1_y - vector2_y) * (vector1_y - vector2_y));
            sumShouder += distance;
          }
          sumShouder = sumShouder / listLeftShouderPoint.size();
          Log.d("BAT", "Shouder Variation: " + sumShouder);

          double sumLeftFoot = 0;
          for(int i = 0; i < listSigleLeftFoot.size() - 1; i++){
            double vector1_x = listSigleLeftFoot.get(i + 1).getPosition().x - listSigleLeftFoot.get(i).getPosition().x;
            double vector1_y = listSigleLeftFoot.get(i + 1).getPosition().y - listSigleLeftFoot.get(i).getPosition().y;
            double distance = Math.sqrt ((vector1_x * vector1_x)
                    + (vector1_y * vector1_y));
            sumLeftFoot += distance;
          }
          sumLeftFoot = sumLeftFoot/listSigleLeftFoot.size();
          Log.d("BAT", "Left Foot Variation: " + sumLeftFoot);

          double sumRightFoot = 0;
          for(int i = 0; i < listSigleRightFoot.size() - 1; i++){
            double vector1_x = listSigleRightFoot.get(i + 1).getPosition().x - listSigleRightFoot.get(i).getPosition().x;
            double vector1_y = listSigleRightFoot.get(i + 1).getPosition().y - listSigleRightFoot.get(i).getPosition().y;
            double distance = Math.sqrt ((vector1_x * vector1_x)
                    + (vector1_y * vector1_y));
            sumRightFoot += distance;
          }
          sumRightFoot = sumRightFoot/listSigleRightFoot.size();
          Log.d("BAT", "Right Foot Variation: " + sumRightFoot);

          ((CameraXLivePreviewActivity)context).left_foot = sumLeftFoot;
          ((CameraXLivePreviewActivity)context).right_foot = sumRightFoot;
          ((CameraXLivePreviewActivity)context).left_hip = getDeveriationOfList(listLeftHipAngle);
          ((CameraXLivePreviewActivity)context).right_hip = getDeveriationOfList(listRightHipAngle);
          ((CameraXLivePreviewActivity)context).shouder = sumShouder;
          ((CameraXLivePreviewActivity)context).FinishProcessing();

          status = IDLE;
          startEvaluation = false;
        }
        break;
      case END_EVAL:

        break;
      default:
        break;
    }
  }
  private double getDeveriationOfList(List<Double> aList){
    double averageValue = 0;
    String strValue = "";
    for(int i = 0; i < aList.size(); i++){
      averageValue += aList.get(i);
      strValue += " "  + aList.get(i);
    }
    averageValue = averageValue/aList.size();
    double averageDevariation = 0;
    for(int i = 0; i < aList.size(); i++){
      averageDevariation += Math.pow(aList.get(i) - averageValue, 2);
    }
    averageDevariation = averageDevariation/aList.size();
    Log.d("BAT", strValue);
    return averageDevariation;
  }

  @Override
  public void stop() {
    super.stop();
    detector.close();
  }

  @Override
  protected Task<PoseWithClassification> detectInImage(InputImage image) {
    return detector
        .process(image)
        .continueWith(
            classificationExecutor,
            task -> {
              Pose pose = task.getResult();
              List<String> classificationResult = new ArrayList<>();
              if (runClassification) {
                if (poseClassifierProcessor == null) {
                  poseClassifierProcessor = new PoseClassifierProcessor(context, isStreamMode);
                }
                classificationResult = poseClassifierProcessor.getPoseResult(pose);
              }
              return new PoseWithClassification(pose, classificationResult);
            });
  }

  @Override
  protected Task<PoseWithClassification> detectInImage(MlImage image) {
    return detector
        .process(image)
        .continueWith(
            classificationExecutor,
            task -> {
              Pose pose = task.getResult();
              List<String> classificationResult = new ArrayList<>();
              if (runClassification) {
                if (poseClassifierProcessor == null) {
                  poseClassifierProcessor = new PoseClassifierProcessor(context, isStreamMode);
                }
                classificationResult = poseClassifierProcessor.getPoseResult(pose);
              }
              return new PoseWithClassification(pose, classificationResult);
            });
  }

  PoseLandmark leftHip, rightHip, leftKnee,rightKnee, leftShouder, rightShouder, leftFoot, rightFoot;
  double angle_left, angle_right, angle_shouder_left, angle_shouder_right, slsLeft, slsRight;

  @Override
  protected void onSuccess(
      @NonNull PoseWithClassification poseWithClassification,
      @NonNull GraphicOverlay graphicOverlay) {
    graphicOverlay.add(
        new PoseGraphic(
            graphicOverlay,
            poseWithClassification.pose,
            showInFrameLikelihood,
            visualizeZ,
            rescaleZForVisualization,
            poseWithClassification.classificationResult));


    //poseWithClassification.pose.getAllPoseLandmarks().get(PoseLandmark.NOSE);
    landmarks = poseWithClassification.pose.getAllPoseLandmarks();
    if (landmarks.isEmpty() == false && landmarks.size() > 30) {
      isBodyDetection = true;
      leftHip = landmarks.get(PoseLandmark.LEFT_HIP);
      rightHip = landmarks.get(PoseLandmark.RIGHT_HIP);
      leftKnee = landmarks.get(PoseLandmark.LEFT_KNEE);
      rightKnee = landmarks.get(PoseLandmark.RIGHT_KNEE);

      leftShouder = landmarks.get(PoseLandmark.LEFT_SHOULDER);
      rightShouder = landmarks.get(PoseLandmark.RIGHT_SHOULDER);

      leftFoot = landmarks.get(PoseLandmark.LEFT_FOOT_INDEX);
      rightFoot = landmarks.get(PoseLandmark.RIGHT_FOOT_INDEX);


      angle_left = calculateAngle(leftHip.getPosition().x, leftHip.getPosition().y, rightHip.getPosition().x, rightHip.getPosition().y, leftKnee.getPosition().x, leftKnee.getPosition().y);
      angle_right = calculateAngle(rightHip.getPosition().x, rightHip.getPosition().y, leftHip.getPosition().x, leftHip.getPosition().y, rightKnee.getPosition().x, rightKnee.getPosition().y);

      angle_shouder_left = calculateAngle(leftShouder.getPosition().x, leftShouder.getPosition().y, rightShouder.getPosition().x, rightShouder.getPosition().y, leftShouder.getPosition().x + 400, leftShouder.getPosition().y);
      angle_shouder_right = calculateAngle(rightShouder.getPosition().x, rightShouder.getPosition().y, leftShouder.getPosition().x, leftShouder.getPosition().y, rightShouder.getPosition().x - 400, rightShouder.getPosition().y);

    }

  }

  private double calculateAngle(double P1X, double P1Y, double P2X, double P2Y,
                                double P3X, double P3Y){

    double numerator = P2Y*(P1X-P3X) + P1Y*(P3X-P2X) + P3Y*(P2X-P1X);
    double denominator = (P2X-P1X)*(P1X-P3X) + (P2Y-P1Y)*(P1Y-P3Y);
    double ratio = numerator/denominator;

    double angleRad = Math.atan(ratio);
    double angleDeg = (angleRad*180)/Math.PI;

    if(angleDeg<0){
      angleDeg = 180+angleDeg;
    }

    return angleDeg;
  }


  @Override
  protected void onFailure(@NonNull Exception e) {
    Log.e(TAG, "Pose detection failed!", e);
  }

  @Override
  protected boolean isMlImageEnabled(Context context) {
    // Use MlImage in Pose Detection by default, change it to OFF to switch to InputImage.
    return true;
  }
}

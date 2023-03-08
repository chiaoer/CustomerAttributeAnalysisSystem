package com.example.mlseriesdemonstrator.object;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import com.example.mlseriesdemonstrator.FaceExtension;
import com.example.mlseriesdemonstrator.R;
import com.example.mlseriesdemonstrator.helpers.MLVideoHelperActivity;
import com.example.mlseriesdemonstrator.helpers.vision.VisionBaseProcessor;
import com.example.mlseriesdemonstrator.helpers.vision.agegenderestimation.AgeGenderEstimationProcessor;
import com.microsoft.azure.sdk.iot.device.ClientOptions;
import com.microsoft.azure.sdk.iot.device.DeviceClient;
import com.microsoft.azure.sdk.iot.device.IotHubClientProtocol;
import com.microsoft.azure.sdk.iot.provisioning.device.AdditionalData;
import com.microsoft.azure.sdk.iot.provisioning.device.ProvisioningDeviceClient;
import com.microsoft.azure.sdk.iot.provisioning.device.ProvisioningDeviceClientRegistrationCallback;
import com.microsoft.azure.sdk.iot.provisioning.device.ProvisioningDeviceClientRegistrationResult;
import com.microsoft.azure.sdk.iot.provisioning.device.ProvisioningDeviceClientStatus;
import com.microsoft.azure.sdk.iot.provisioning.device.ProvisioningDeviceClientTransportProtocol;
import com.microsoft.azure.sdk.iot.provisioning.device.internal.exceptions.ProvisioningDeviceClientException;
import com.microsoft.azure.sdk.iot.provisioning.security.SecurityProviderSymmetricKey;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;

public class VisitorAnalysisActivity extends MLVideoHelperActivity implements AgeGenderEstimationProcessor.AgeGenderCallback {

    private Interpreter ageModelInterpreter;
    private Interpreter genderModelInterpreter;

    private int facesCount;
    private int smilingCount;
    private int maleCount;
    private int femaleCount;
    private int kidsCount;
    private int youngCount;
    private int adultCount;
    private int agedCount;

    private Set<Integer> faceTrackingIdSet = new HashSet<>();
    private static final String TAG = "VisitorAnalysisActivity";

    /************ Azure IoT Connection Parameters ***********/
    private static final String deviceConnectionString = System.getenv("IOTHUB_DEVICE_CONNECTION_STRING");
    private static final String deviceSecurityType = "DPS";
    private static final String MODEL_ID = "dtmi:Synnex:DoorwayDevice;1";
    private static final String scopeId = System.getenv("IOTHUB_DEVICE_DPS_ID_SCOPE");
    private static final String globalEndpoint = System.getenv("IOTHUB_DEVICE_DPS_ENDPOINT");//synnex service
    private static final String deviceSymmetricKey = System.getenv("IOTHUB_DEVICE_DPS_DEVICE_KEY");
    private static final String registrationId = System.getenv("IOTHUB_DEVICE_DPS_DEVICE_ID");

    // Plug and play features are available over MQTT, MQTT_WS, AMQPS, and AMQPS_WS.
    private static final ProvisioningDeviceClientTransportProtocol provisioningProtocol = ProvisioningDeviceClientTransportProtocol.MQTT;
    private static final IotHubClientProtocol protocol = IotHubClientProtocol.MQTT;

    private static final int MAX_TIME_TO_WAIT_FOR_REGISTRATION = 1000; // in milli seconds

    private static DeviceClient deviceClient;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //Initialize device client instance.
        new Thread(new Runnable() {
            public void run() {
                // TODO Auto-generated method stub
                // 您要在執行緒作的事
                try {
                    InitClient();
                } catch (Exception e2)
                {
                    Log.d(TAG, "Exception while opening IoTHub connection: " + e2.toString());
                }            }
        }).start();
    }

    @Override
    protected VisionBaseProcessor setProcessor() {
        try {
            ageModelInterpreter = new Interpreter(FileUtil.loadMappedFile(this, "model_lite_age_q.tflite"), new Interpreter.Options());
            genderModelInterpreter = new Interpreter(FileUtil.loadMappedFile(this, "model_lite_gender_q.tflite"), new Interpreter.Options());
        } catch (IOException e) {
            e.printStackTrace();
        }

        AgeGenderEstimationProcessor ageGenderEstimationProcessor = new AgeGenderEstimationProcessor(
                ageModelInterpreter,
                genderModelInterpreter,
                graphicOverlay,
                this
        );
        ageGenderEstimationProcessor.activity = this;
        return ageGenderEstimationProcessor;
    }

    public void setTestImage(Bitmap cropToBBox) {
        if (cropToBBox == null) {
            return;
        }
        runOnUiThread(() -> ((ImageView) findViewById(R.id.testImageView)).setImageBitmap(cropToBBox));
    }

    @Override
    public void onFaceDetected(FaceExtension face, int age, int gender) {
        Log.d(TAG, "## onFaceDetected()...faceTrackingIdSet size = " + faceTrackingIdSet.size() + ", facesCount = " + facesCount);

        if (!faceTrackingIdSet.contains(face.faceOri.getTrackingId())) {
            facesCount++;

            if (face.faceOri.getSmilingProbability() != null && face.faceOri.getSmilingProbability() > .79f) {
                smilingCount++;
            }

            if (age < 12) {
                kidsCount++;
            } else if (age < 20) {
                youngCount++;
            } else if (age < 60) {
                adultCount++;
            } else {
                agedCount++;
            }

            if (gender == 0) {
                maleCount++;
            } else {
                femaleCount++;
            }

            StringBuilder builder = new StringBuilder();
            builder.append("Total faces: ").append(facesCount).append(", Smiling: ").append((int) ((smilingCount/(float) facesCount) * 100.0f)).append("%\n")
                    .append("Male: ").append(maleCount).append(", Female: ").append(femaleCount).append("\n")
                    .append("Kids: ").append(kidsCount).append(", Young: ").append(youngCount).append("\n")
                    .append("Adults: ").append(adultCount).append(", Aged: ").append(agedCount);

            setOutputText(builder.toString());

            faceTrackingIdSet.add(face.faceOri.getTrackingId());//Fix original bug to add new faces
        }
    }

    /*************************** Azure IoT Connection Functions **************************/
    private void InitClient() throws URISyntaxException, IOException, ProvisioningDeviceClientException, InterruptedException {
        // This environment variable indicates if DPS or IoT Hub connection string will be used to provision the device.
        // Expected values: (case-insensitive)
        // "DPS" - The sample will use DPS to provision the device.
        // "connectionString" - The sample will use IoT Hub connection string to provision the device.
        if ((deviceSecurityType == null) || deviceSecurityType.isEmpty())
        {
            throw new IllegalArgumentException("Device security type needs to be specified, please set the environment variable \"IOTHUB_DEVICE_SECURITY_TYPE\"");
        }

        Log.d(TAG, "Initialize the device client.");

        switch (deviceSecurityType.toLowerCase())
        {
            case "dps":
            {
                if (validateArgsForDpsFlow())
                {
                    initializeAndProvisionDevice();
                    break;
                }
                throw new IllegalArgumentException("Required environment variables are not set for DPS flow, please recheck your environment.");
            }
            case "connectionstring":
            {
                if (validateArgsForIotHubFlow())
                {
                    initializeDeviceClient();
                    break;
                }
                throw new IllegalArgumentException("Required environment variables are not set for IoT Hub flow, please recheck your environment.");
            }
            default:
            {
                throw new IllegalArgumentException("Unrecognized value for IOTHUB_DEVICE_SECURITY_TYPE received: {s_deviceSecurityType}." +
                        " It should be either \"DPS\" or \"connectionString\" (case-insensitive).");
            }
        }
    }
    private static boolean validateArgsForDpsFlow()
    {
        return !((globalEndpoint == null || globalEndpoint.isEmpty())
                && (scopeId == null || scopeId.isEmpty())
                && (registrationId == null || registrationId.isEmpty())
                && (deviceSymmetricKey == null || deviceSymmetricKey.isEmpty()));
    }
    private static boolean validateArgsForIotHubFlow()
    {
        return !(deviceConnectionString == null || deviceConnectionString.isEmpty());
    }
    /**
     * Initialize the device client instance over Mqtt protocol, setting the ModelId into ClientOptions.
     * This method also sets a connection status change callback, that will get triggered any time the device's connection status changes.
     */
    private static void initializeDeviceClient() throws URISyntaxException, IOException {
        ClientOptions options = new ClientOptions();
        options.setModelId(MODEL_ID);
        deviceClient = new DeviceClient(deviceConnectionString, protocol, options);

        deviceClient.registerConnectionStatusChangeCallback((status, statusChangeReason, throwable, callbackContext) -> {
            Log.d(TAG, "Connection status change registered: status={}, reason={}" + status + "," + statusChangeReason);

            if (throwable != null) {
                Log.d(TAG, "The connection status change was caused by the following Throwable: {}" + throwable.getMessage());
                throwable.printStackTrace();
            }
        }, deviceClient);

        try {
            deviceClient.open();
        } catch (Exception e2) {
            Log.e(TAG, "Exception while opening IoTHub connection: " + e2.getMessage());
            deviceClient.closeNow();
            Log.e(TAG, "Shutting down...");
        }
    }
    private static void initializeAndProvisionDevice() throws ProvisioningDeviceClientException, IOException, URISyntaxException, InterruptedException {
        SecurityProviderSymmetricKey securityClientSymmetricKey = new SecurityProviderSymmetricKey(deviceSymmetricKey.getBytes(StandardCharsets.UTF_8), registrationId);
        ProvisioningDeviceClient provisioningDeviceClient;
        ProvisioningStatus provisioningStatus = new ProvisioningStatus();

        provisioningDeviceClient = ProvisioningDeviceClient.create(globalEndpoint, scopeId, provisioningProtocol, securityClientSymmetricKey);

        AdditionalData additionalData = new AdditionalData();
        additionalData.setProvisioningPayload(com.microsoft.azure.sdk.iot.provisioning.device.plugandplay.PnpHelper.createDpsPayload(MODEL_ID));

        provisioningDeviceClient.registerDevice(new ProvisioningDeviceClientRegistrationCallbackImpl(), provisioningStatus, additionalData);

        while (provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getProvisioningDeviceClientStatus() != ProvisioningDeviceClientStatus.PROVISIONING_DEVICE_STATUS_ASSIGNED)
        {
            if (provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getProvisioningDeviceClientStatus() == ProvisioningDeviceClientStatus.PROVISIONING_DEVICE_STATUS_ERROR ||
                    provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getProvisioningDeviceClientStatus() == ProvisioningDeviceClientStatus.PROVISIONING_DEVICE_STATUS_DISABLED ||
                    provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getProvisioningDeviceClientStatus() == ProvisioningDeviceClientStatus.PROVISIONING_DEVICE_STATUS_FAILED)
            {
                provisioningStatus.exception.printStackTrace();
                Log.d(TAG, "Registration error, bailing out");
                break;
            }
            Log.d(TAG, "Waiting for Provisioning Service to register");
            Thread.sleep(MAX_TIME_TO_WAIT_FOR_REGISTRATION);
        }

        ClientOptions options = new ClientOptions();
        options.setModelId(MODEL_ID);

        try {
            if (provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getProvisioningDeviceClientStatus() == ProvisioningDeviceClientStatus.PROVISIONING_DEVICE_STATUS_ASSIGNED) {
                Log.d(TAG, "IotHUb Uri : " + provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getIothubUri());
                Log.d(TAG, "Device ID : " + provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getDeviceId());

                String iotHubUri = provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getIothubUri();
                String deviceId = provisioningStatus.provisioningDeviceClientRegistrationInfoClient.getDeviceId();

                Log.d(TAG, "Opening the device client.");

                deviceClient = DeviceClient.createFromSecurityProvider(iotHubUri, deviceId, securityClientSymmetricKey, IotHubClientProtocol.MQTT, options);
                deviceClient.open();
            }
        } catch (Exception e2) {
            Log.e(TAG, "Exception while opening IoTHub connection: " + e2.getMessage());
            deviceClient.closeNow();
            Log.e(TAG, "Shutting down...");
        }
    }
    static class ProvisioningStatus
    {
        ProvisioningDeviceClientRegistrationResult provisioningDeviceClientRegistrationInfoClient = new ProvisioningDeviceClientRegistrationResult();
        Exception exception;
    }
    static class ProvisioningDeviceClientRegistrationCallbackImpl implements ProvisioningDeviceClientRegistrationCallback
    {
        @Override
        public void run(ProvisioningDeviceClientRegistrationResult provisioningDeviceClientRegistrationResult, Exception exception, Object context)
        {
            if (context instanceof ProvisioningStatus)
            {
                ProvisioningStatus status = (ProvisioningStatus) context;
                status.provisioningDeviceClientRegistrationInfoClient = provisioningDeviceClientRegistrationResult;
                status.exception = exception;
            }
            else
            {
                Log.d(TAG, "Received unknown context");
            }
        }
    }
}

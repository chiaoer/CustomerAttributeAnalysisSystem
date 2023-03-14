package com.example.mlseriesdemonstrator.object;

import android.app.ActivityManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.StatFs;
import android.provider.Settings;
import android.util.Log;
import android.widget.ImageView;

import com.example.mlseriesdemonstrator.FaceExtension;
import com.example.mlseriesdemonstrator.PnpConvention;
import com.example.mlseriesdemonstrator.R;
import com.example.mlseriesdemonstrator.helpers.MLVideoHelperActivity;
import com.example.mlseriesdemonstrator.helpers.vision.VisionBaseProcessor;
import com.example.mlseriesdemonstrator.helpers.vision.agegenderestimation.AgeGenderEstimationProcessor;
import com.microsoft.azure.sdk.iot.device.ClientOptions;
import com.microsoft.azure.sdk.iot.device.DeviceClient;
import com.microsoft.azure.sdk.iot.device.DeviceTwin.Property;
import com.microsoft.azure.sdk.iot.device.DeviceTwin.TwinPropertyCallBack;
import com.microsoft.azure.sdk.iot.device.IotHubClientProtocol;
import com.microsoft.azure.sdk.iot.device.IotHubEventCallback;
import com.microsoft.azure.sdk.iot.device.IotHubStatusCode;
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

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import lombok.NonNull;

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

    //Device information parameters
    private String hostname;
    private String cpuInfo;
    private long cpuCores;
    private long cpuMaxfreq;
    private String baseboardManufacturer;
    private String baseboardSerialNumber;
    private String osVersion;
    private String osBuildNumber;
    private long memTotal;
    private long logicalDISKtotal;
    private String ipLocal;
    private String ipPublic;
    private double highTemp;
    private double currentTempGPU;
    private double cpuClock;
    private long memFree;
    private long memUsage;
    private long logicalDISKfree;
    private long logicalDISKusage;
    private double currentTemp;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        readPropertyValue();

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
                }
                updateReportedProperties();
            }
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

        String componentName = "FaceAttributeDetect";

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

            /*************************** Send msg to Azure **************************/
            componentName = "FaceAttributeDetect";

            Map<String, Object> faceAttribute = new HashMap<>();
            faceAttribute.put("people_count",facesCount);
            faceAttribute.put("age",face.ga_result.age);
            faceAttribute.put("gender",face.ga_result.gender);

            Log.d(TAG, "## Send msg to Azure...faceAttribute = " + faceAttribute);
            com.microsoft.azure.sdk.iot.device.Message message = PnpConvention.createIotHubMessageUtf8(faceAttribute, componentName);
            deviceClient.sendEventAsync(message, new MessageIotHubEventCallback(), message);
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
    /**
     * The callback to be invoked when a telemetry response is received from IoT Hub.
     */
    private static class MessageIotHubEventCallback implements IotHubEventCallback {

        @Override
        public void execute(IotHubStatusCode responseStatus, Object callbackContext) {
            com.microsoft.azure.sdk.iot.device.Message msg = (com.microsoft.azure.sdk.iot.device.Message) callbackContext;
            Log.d(TAG, "Telemetry - Response from IoT Hub: message Id={}, status={}" + msg.getMessageId() + "," + responseStatus.name());
        }
    }
    /********************* Send AndroidDeviceInfo1 information to the cloud. ********************/
    /**
     * The callback to be invoked in response to device twin operations in IoT Hub.
     */
    private static class TwinIotHubEventCallback implements IotHubEventCallback {

        @Override
        public void execute(IotHubStatusCode responseStatus, Object callbackContext) {
            Log.d(TAG,"Property - Response from IoT Hub: {}" + responseStatus.name());
        }
    }

    /**
     * The callback to be invoked for a property change that is not explicitly monitored by the device.
     */
    private static class GenericPropertyUpdateCallback implements TwinPropertyCallBack {

        @Override
        public void TwinPropertyCallBack(Property property, Object context) {
            Log.d(TAG, "Property - Received property unhandled by device, key={}, value={}" + property.getKey() + ", " + property.getValue());
        }
    }
    private void readPropertyValue() {
        hostname = getHostname();
        cpuInfo = getCPUInfo();
        cpuCores = getNumberOfCores();
        cpuMaxfreq = getCPUMaxFreq();
        baseboardManufacturer = getManufacturer();
        baseboardSerialNumber = getModel();
        osVersion = getAndroidVersion();
        osBuildNumber = getAndroidBuildNumber();
        memTotal = getMEMTotal();
        logicalDISKtotal = getTotalDiskSpace();
        ipLocal = getLocalIpAddress();
        ipPublic = getPublicIPAddress();
        highTemp = getCpuTemp();
        currentTempGPU = getGpuTemp();
        cpuClock = getCPUFreq();
        memFree = getMEMavail();
        memUsage = getMEMusage();
        logicalDISKfree = freeDISK();
        logicalDISKusage = busyDISK();
        currentTemp = getCpuTemp();

        Log.d(TAG,"/*********** Properties dump ************/");
        Log.d(TAG,"hostname = " + hostname);
        Log.d(TAG,"cpuInfo = " + cpuInfo);
        Log.d(TAG,"cpuCores = " + cpuCores);
        Log.d(TAG,"cpuMaxfreq = " + cpuMaxfreq + "GHz");
        Log.d(TAG,"baseboardManufacturer = " + baseboardManufacturer);
        Log.d(TAG,"baseboardSerialNumber = " + baseboardSerialNumber);
        Log.d(TAG,"osVersion = " + osVersion);
        Log.d(TAG,"osBuildNumber = " + osBuildNumber);
        Log.d(TAG,"memTotal = " + memTotal + "MB");
        Log.d(TAG,"logicalDISKtotal = " + logicalDISKtotal + "MB");
        Log.d(TAG,"ipLocal = " + ipLocal);
        Log.d(TAG,"ipPublic = " + ipPublic);
        Log.d(TAG,"highTemp = " + highTemp);
        Log.d(TAG,"currentTempGPU = " + currentTempGPU);
        Log.d(TAG,"cpuClock = " + cpuClock);
        Log.d(TAG,"memFree = " + memFree + "MB");
        Log.d(TAG,"memUsage = " + memUsage + "MB");
        Log.d(TAG,"logicalDISKfree = " + logicalDISKfree + "MB");
        Log.d(TAG,"logicalDISKusage = " + logicalDISKusage + "MB");
        Log.d(TAG,"currentTemp = " + currentTemp);
    }
    private void updateReportedProperties() {
        String componentName = "AndroidDeviceInfo1";

        Set<Property> reportProperties = PnpConvention.createComponentPropertyPatch(componentName, new HashMap<String, Object>()
        {{
            put("hostname", hostname);
            put("cpuInfo", cpuInfo);
            put("cpuCores", cpuCores);
            put("cpuMaxfreq", cpuMaxfreq);
            put("baseboardManufacturer", baseboardManufacturer);
            put("baseboardSerialNumber", baseboardSerialNumber);
            put("osVersion", osVersion);
            put("osBuildNumber", osBuildNumber);
            put("memTotal", memTotal);
            put("logicalDISKtotal", logicalDISKtotal);
            put("ipLocal", ipLocal);
            put("ipPublic", ipPublic);
            put("highTemp", highTemp);
        }});

        try {
            deviceClient.startDeviceTwin(new TwinIotHubEventCallback(), null, new GenericPropertyUpdateCallback(), null);
            deviceClient.sendReportedProperties(reportProperties);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d(TAG,"updateReportedProperties()....sendReportedProperties");
    }
    private String getHostname() {
        return Settings.Global.getString(getApplicationContext().getContentResolver(), Settings.Global.DEVICE_NAME);
    }

    private String getManufacturer() {
        return Build.MANUFACTURER;
    }

    private String getModel() {
        return Build.MODEL;
    }

    @NonNull
    private String getAndroidVersion() {
        return "android" + Build.VERSION.RELEASE;
    }

    @NonNull
    private String getAndroidBuildNumber() {
        return Integer.toString(Build.VERSION.SDK_INT);
    }

    private String getCPUInfo() {
        String str, output = "Unavailable";
        BufferedReader br;

        try {
            br = new BufferedReader(new FileReader("/proc/cpuinfo"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return "Unavailable";
        }

        try{
            while((str = br.readLine()) != null) {
                String[] data = str.split(":");
                if (data.length > 1) {
                    String key = data[0].trim().replace(" ", "_");
                    if (key.equals("Hardware") || key.equals("model_name")) {
                        output = data[1].trim();
                    }
                }
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return output;
    }
    private long getNumberOfCores() {
        return Runtime.getRuntime().availableProcessors();
    }

    private long getCPUMaxFreq() {
        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return 0;
        }

        try {
            String cpuMaxFreq = reader.readLine();
            reader.close();
            return Long.parseLong(cpuMaxFreq) / 1000000L;
        } catch (IOException e) {
            e.printStackTrace();
            return 0;
        }
    }

    private long getMEMTotal() {
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        activityManager.getMemoryInfo(mi);

        return mi.totalMem / 0x100000L;
    }

    private long getTotalDiskSpace() {
        StatFs statFs = new StatFs(Environment.getRootDirectory().getAbsolutePath());
        long rootDiskSpace = statFs.getBlockCountLong() * statFs.getBlockSizeLong();
        statFs = new StatFs(Environment.getDataDirectory().getAbsolutePath());
        long dataDiskSpace = statFs.getBlockCountLong() * statFs.getBlockSizeLong();

        return (rootDiskSpace + dataDiskSpace) / 0x100000L;
    }
    private String getLocalIpAddress() {
        try {
            for (Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface intf = en.nextElement();
                for (Enumeration<InetAddress> enumIpAddr = intf.getInetAddresses(); enumIpAddr.hasMoreElements();) {
                    InetAddress inetAddress = enumIpAddr.nextElement();
                    if (!inetAddress.isLoopbackAddress() && inetAddress instanceof Inet4Address) {
                        return inetAddress.getHostAddress();
                    }
                }
            }
        } catch (SocketException e) {
            e.printStackTrace();
            return "Unavailable";
        }

        return "Unavailable";
    }

    private String getPublicIPAddress() {
        ExecutorService es = Executors.newSingleThreadExecutor();
        Future<String> result = es.submit(new Callable<String>() {
            public String call() throws Exception {
                try {
                    URL url = new URL("http://whatismyip.akamai.com/");
                    HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
                    try {
                        InputStream in = new BufferedInputStream(urlConnection.getInputStream());
                        BufferedReader r = new BufferedReader(new InputStreamReader(in));
                        StringBuilder total = new StringBuilder();
                        String line;
                        while ((line = r.readLine()) != null) {
                            total.append(line).append('\n');
                        }
                        urlConnection.disconnect();
                        return total.toString();
                    }finally {
                        urlConnection.disconnect();
                    }
                }catch (IOException e){
                    Log.e("Public IP: ",e.getMessage());
                }
                return "Unavailable";
            }
        });

        try {
            return result.get();
        } catch (Exception e) {
            e.printStackTrace();
            return "Unavailable";
        }
    }

    private double getCpuTemp() {
        try {
            BufferedReader reader = new BufferedReader(new FileReader("/sys/devices/virtual/thermal/thermal_zone0/temp"));
            String cputemp = reader.readLine();
            reader.close();
            return Double.parseDouble(cputemp) / 1000;
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    private double getGpuTemp() {
        try {
            BufferedReader reader = new BufferedReader(new FileReader("/sys/class/thermal/thermal_zone10/temp"));
            String gputemp = reader.readLine();
            reader.close();
            return Double.parseDouble(gputemp) / 1000;
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    private double getCPUFreq() {
        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return 0;
        }

        try {
            String cpuFreq = reader.readLine();
            reader.close();
            return Double.parseDouble(cpuFreq) / 1000000;
        } catch (IOException e) {
            e.printStackTrace();
            return 0;
        }
    }
    private long getMEMavail() {
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        activityManager.getMemoryInfo(mi);

        return mi.availMem / 0x100000L;
    }
    private long getMEMusage() {
        return getMEMTotal() - getMEMavail();
    }
    private long freeDISK()
    {
        StatFs statFs = new StatFs(Environment.getRootDirectory().getAbsolutePath());
        long freeRoot = (statFs.getAvailableBlocksLong() * statFs.getBlockSizeLong());
        statFs = new StatFs(Environment.getDataDirectory().getAbsolutePath());
        long freeData = statFs.getAvailableBlocksLong() * statFs.getBlockSizeLong();

        return (freeRoot + freeData) / 0x100000L;
    }
    private long busyDISK()
    {
        return getTotalDiskSpace() - freeDISK();
    }

}

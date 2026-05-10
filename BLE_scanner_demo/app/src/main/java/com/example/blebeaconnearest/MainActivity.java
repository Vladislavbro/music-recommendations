package com.example.blebeaconnearest;

import android.Manifest;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class MainActivity extends Activity {
    private static final int REQUEST_PERMISSIONS = 42;
    private static final long NEAREST_TIMEOUT_MS = 5000L;
    private static final long FORGET_AFTER_MS = 15000L;
    private static final String PREFS_NAME = "saved_ble_beacons";
    private static final String PREFS_BEACONS = "beacons";

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final List<SavedBeacon> savedBeacons = new ArrayList<>();
    private final Map<String, LiveBeacon> liveBeacons = new HashMap<>();
    private final int[] palette = {
            Color.rgb(239, 83, 80),
            Color.rgb(255, 193, 7),
            Color.rgb(102, 187, 106),
            Color.rgb(38, 166, 154),
            Color.rgb(66, 133, 244),
            Color.rgb(171, 71, 188)
    };

    private BluetoothAdapter bluetoothAdapter;
    private BluetoothLeScanner bleScanner;
    private boolean isScanning;
    private int selectedColor;

    private LinearLayout livePanel;
    private TextView liveTitle;
    private TextView liveSubtitle;
    private EditText macInput;
    private EditText nameInput;
    private LinearLayout colorRow;
    private LinearLayout savedList;
    private TextView statusView;
    private Button permissionButton;

    private final Runnable tickRunnable = new Runnable() {
        @Override
        public void run() {
            pruneOldSignals();
            updateNearestPanel();
            if (isScanning) {
                handler.postDelayed(this, 1000L);
            }
        }
    };

    private final ScanCallback scanCallback = new ScanCallback() {
        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            handleScanResult(result);
        }

        @Override
        public void onBatchScanResults(List<ScanResult> results) {
            for (ScanResult result : results) {
                handleScanResult(result);
            }
        }

        @Override
        public void onScanFailed(int errorCode) {
            handler.post(() -> {
                isScanning = false;
                statusView.setText("Ошибка BLE-сканирования: " + errorCode);
                updateNearestPanel();
            });
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        selectedColor = palette[4];
        loadSavedBeacons();
        buildUi();
        renderSavedList();
        updatePermissionUi();
        setupBluetooth();

        if (hasRequiredPermissions()) {
            startScan();
        } else {
            requestRequiredPermissions();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (hasRequiredPermissions()) {
            startScan();
        }
    }

    @Override
    protected void onPause() {
        stopScan();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        stopScan();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSIONS) {
            updatePermissionUi();
            if (hasRequiredPermissions()) {
                startScan();
            } else {
                statusView.setText("Разрешите Bluetooth и геолокацию, чтобы сканировать BLE-маяки.");
                setEmptyPanel("Нет доступа к сканированию", "Ожидаю разрешения Bluetooth и геолокации");
            }
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(247, 247, 242));
        root.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));

        livePanel = new LinearLayout(this);
        livePanel.setOrientation(LinearLayout.VERTICAL);
        livePanel.setGravity(Gravity.CENTER_VERTICAL);
        livePanel.setPadding(dp(22), dp(18), dp(22), dp(18));
        LinearLayout.LayoutParams liveParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(210));
        liveParams.setMargins(dp(16), dp(16), dp(16), dp(10));
        root.addView(livePanel, liveParams);

        liveTitle = new TextView(this);
        liveTitle.setTextSize(30);
        liveTitle.setTypeface(Typeface.DEFAULT_BOLD);
        liveTitle.setGravity(Gravity.CENTER);
        liveTitle.setIncludeFontPadding(false);
        livePanel.addView(liveTitle, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        liveSubtitle = new TextView(this);
        liveSubtitle.setTextSize(15);
        liveSubtitle.setGravity(Gravity.CENTER);
        liveSubtitle.setPadding(0, dp(12), 0, 0);
        livePanel.addView(liveSubtitle, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        ScrollView scrollView = new ScrollView(this);
        scrollView.setFillViewport(false);
        root.addView(scrollView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f));

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(dp(16), dp(6), dp(16), dp(28));
        scrollView.addView(content, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        statusView = label("Подготовка сканера", 14, Color.rgb(70, 74, 80), Typeface.NORMAL);
        statusView.setPadding(0, 0, 0, dp(10));
        content.addView(statusView);

        permissionButton = new Button(this);
        permissionButton.setText("Разрешить доступ");
        permissionButton.setAllCaps(false);
        permissionButton.setVisibility(View.GONE);
        permissionButton.setOnClickListener(v -> requestRequiredPermissions());
        content.addView(permissionButton, matchWrapParams(0, dp(8)));

        TextView formTitle = label("Сохранить маяк", 22, Color.rgb(22, 26, 30), Typeface.BOLD);
        formTitle.setPadding(0, dp(8), 0, dp(12));
        content.addView(formTitle);

        macInput = new EditText(this);
        macInput.setHint("MAC адрес: AA:BB:CC:DD:EE:FF");
        macInput.setSingleLine(true);
        macInput.setInputType(InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS);
        content.addView(macInput, matchWrapParams(0, dp(10)));

        nameInput = new EditText(this);
        nameInput.setHint("Название маяка");
        nameInput.setSingleLine(true);
        content.addView(nameInput, matchWrapParams(0, dp(14)));

        colorRow = new LinearLayout(this);
        colorRow.setOrientation(LinearLayout.HORIZONTAL);
        colorRow.setGravity(Gravity.CENTER_VERTICAL);
        content.addView(colorRow, matchWrapParams(0, dp(16)));
        renderColorSwatches();

        Button saveButton = new Button(this);
        saveButton.setText("Сохранить маяк");
        saveButton.setAllCaps(false);
        saveButton.setTextSize(16);
        saveButton.setOnClickListener(v -> saveBeaconFromForm());
        content.addView(saveButton, matchWrapParams(0, dp(16)));

        TextView listTitle = label("Сохраненные маяки", 22, Color.rgb(22, 26, 30), Typeface.BOLD);
        listTitle.setPadding(0, dp(8), 0, dp(10));
        content.addView(listTitle);

        savedList = new LinearLayout(this);
        savedList.setOrientation(LinearLayout.VERTICAL);
        content.addView(savedList, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        setContentView(root);
        updateNearestPanel();
    }

    private void renderColorSwatches() {
        colorRow.removeAllViews();
        for (int color : palette) {
            TextView swatch = new TextView(this);
            swatch.setContentDescription("Выбрать цвет");
            swatch.setBackground(swatchBackground(color, color == selectedColor));
            swatch.setOnClickListener(v -> {
                selectedColor = color;
                renderColorSwatches();
            });
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(42), dp(42));
            params.setMargins(0, 0, dp(10), 0);
            colorRow.addView(swatch, params);
        }
    }

    private void saveBeaconFromForm() {
        String mac = normalizeMac(macInput.getText().toString());
        String name = nameInput.getText().toString().trim();

        if (mac == null) {
            statusView.setText("Проверьте MAC адрес маяка.");
            return;
        }
        if (name.isEmpty()) {
            statusView.setText("Введите название маяка.");
            return;
        }

        SavedBeacon existing = findSaved(mac);
        if (existing == null) {
            savedBeacons.add(new SavedBeacon(mac, name, selectedColor));
            statusView.setText("Маяк сохранен.");
        } else {
            existing.name = name;
            existing.color = selectedColor;
            statusView.setText("Маяк обновлен.");
        }

        saveBeacons();
        macInput.setText("");
        nameInput.setText("");
        renderSavedList();
        updateNearestPanel();

        if (!hasRequiredPermissions()) {
            requestRequiredPermissions();
        } else {
            startScan();
        }
    }

    private void renderSavedList() {
        savedList.removeAllViews();
        if (savedBeacons.isEmpty()) {
            TextView empty = label("Пока нет сохраненных маяков", 15, Color.rgb(92, 96, 102), Typeface.NORMAL);
            empty.setPadding(0, dp(4), 0, dp(10));
            savedList.addView(empty);
            return;
        }

        for (SavedBeacon beacon : savedBeacons) {
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(dp(12), dp(10), dp(10), dp(10));
            row.setBackground(rounded(Color.WHITE, dp(8), Color.rgb(224, 226, 220), 1));

            TextView chip = new TextView(this);
            chip.setBackground(swatchBackground(beacon.color, false));
            row.addView(chip, new LinearLayout.LayoutParams(dp(28), dp(28)));

            LinearLayout textColumn = new LinearLayout(this);
            textColumn.setOrientation(LinearLayout.VERTICAL);
            textColumn.setPadding(dp(12), 0, dp(10), 0);

            TextView name = label(beacon.name, 17, Color.rgb(25, 29, 33), Typeface.BOLD);
            TextView mac = label(beacon.mac, 13, Color.rgb(94, 98, 104), Typeface.NORMAL);
            textColumn.addView(name);
            textColumn.addView(mac);
            row.addView(textColumn, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

            Button remove = new Button(this);
            remove.setText("Удалить");
            remove.setAllCaps(false);
            remove.setOnClickListener(v -> {
                savedBeacons.remove(beacon);
                saveBeacons();
                renderSavedList();
                updateNearestPanel();
            });
            row.addView(remove, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT));

            LinearLayout.LayoutParams params = matchWrapParams(0, dp(10));
            savedList.addView(row, params);
        }
    }

    private void setupBluetooth() {
        BluetoothManager manager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
        if (manager != null) {
            bluetoothAdapter = manager.getAdapter();
        }
    }

    private void startScan() {
        updatePermissionUi();
        if (isScanning) {
            updateNearestPanel();
            return;
        }
        if (!hasRequiredPermissions()) {
            statusView.setText("Нужны разрешения для BLE-сканирования.");
            setEmptyPanel("Нет доступа к сканированию", "Ожидаю разрешения Bluetooth и геолокации");
            return;
        }
        if (bluetoothAdapter == null) {
            setupBluetooth();
        }
        if (bluetoothAdapter == null) {
            statusView.setText("Bluetooth LE недоступен на устройстве.");
            setEmptyPanel("BLE недоступен", "На этом устройстве нет Bluetooth LE");
            return;
        }

        try {
            if (!bluetoothAdapter.isEnabled()) {
                statusView.setText("Включите Bluetooth на телефоне.");
                setEmptyPanel("Bluetooth выключен", "Жду включения Bluetooth");
                return;
            }
            bleScanner = bluetoothAdapter.getBluetoothLeScanner();
            if (bleScanner == null) {
                statusView.setText("BLE-сканер недоступен.");
                setEmptyPanel("Сканер недоступен", "Проверьте Bluetooth на телефоне");
                return;
            }

            ScanSettings settings = new ScanSettings.Builder()
                    .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
                    .setReportDelay(0L)
                    .build();
            bleScanner.startScan(null, settings, scanCallback);
            isScanning = true;
            statusView.setText("Сканирование BLE активно.");
            handler.removeCallbacks(tickRunnable);
            handler.post(tickRunnable);
            updateNearestPanel();
        } catch (SecurityException exception) {
            statusView.setText("Нет разрешения на BLE-сканирование.");
            setEmptyPanel("Нет доступа к сканированию", "Ожидаю разрешения Bluetooth и геолокации");
        } catch (IllegalStateException exception) {
            statusView.setText("Не удалось запустить BLE-сканирование.");
            setEmptyPanel("Сканер не запущен", "Попробуйте включить Bluetooth заново");
        }
    }

    private void stopScan() {
        handler.removeCallbacks(tickRunnable);
        if (!isScanning || bleScanner == null) {
            isScanning = false;
            return;
        }
        try {
            bleScanner.stopScan(scanCallback);
        } catch (SecurityException ignored) {
            // Permission can be revoked while the Activity is active.
        } finally {
            isScanning = false;
        }
    }

    private void handleScanResult(ScanResult result) {
        if (result == null || result.getDevice() == null) {
            return;
        }

        String mac;
        try {
            mac = normalizeMac(result.getDevice().getAddress());
        } catch (SecurityException exception) {
            return;
        }
        if (mac == null) {
            return;
        }

        int rssi = result.getRssi();
        handler.post(() -> {
            liveBeacons.put(mac, new LiveBeacon(mac, rssi, System.currentTimeMillis()));
            updateNearestPanel();
        });
    }

    private void updateNearestPanel() {
        long now = System.currentTimeMillis();
        SavedBeacon nearest = null;
        LiveBeacon nearestSignal = null;

        for (SavedBeacon saved : savedBeacons) {
            LiveBeacon signal = liveBeacons.get(saved.mac);
            if (signal == null || now - signal.lastSeenMs > NEAREST_TIMEOUT_MS) {
                continue;
            }
            if (nearestSignal == null || signal.rssi > nearestSignal.rssi) {
                nearest = saved;
                nearestSignal = signal;
            }
        }

        if (nearest != null && nearestSignal != null) {
            int textColor = readableTextColor(nearest.color);
            livePanel.setBackground(rounded(nearest.color, dp(8), Color.TRANSPARENT, 0));
            liveTitle.setText(nearest.name);
            liveTitle.setTextColor(textColor);
            liveSubtitle.setText(String.format(
                    Locale.US,
                    "%s   RSSI %d dBm",
                    nearest.mac,
                    nearestSignal.rssi));
            liveSubtitle.setTextColor(textColor);
            return;
        }

        if (savedBeacons.isEmpty()) {
            setEmptyPanel("Добавьте BLE маяк", "Сканирование начнется после выдачи разрешений");
        } else if (isScanning) {
            setEmptyPanel("Маяк не найден", "Жду свежий сигнал сохраненного маяка");
        } else {
            setEmptyPanel("Сканирование остановлено", "Откройте приложение и включите Bluetooth");
        }
    }

    private void setEmptyPanel(String title, String subtitle) {
        livePanel.setBackground(rounded(Color.rgb(232, 234, 228), dp(8), Color.TRANSPARENT, 0));
        liveTitle.setText(title);
        liveTitle.setTextColor(Color.rgb(32, 36, 40));
        liveSubtitle.setText(subtitle);
        liveSubtitle.setTextColor(Color.rgb(87, 92, 98));
    }

    private void pruneOldSignals() {
        long now = System.currentTimeMillis();
        Iterator<Map.Entry<String, LiveBeacon>> iterator = liveBeacons.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, LiveBeacon> entry = iterator.next();
            if (now - entry.getValue().lastSeenMs > FORGET_AFTER_MS) {
                iterator.remove();
            }
        }
    }

    private boolean hasRequiredPermissions() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
            return true;
        }
        for (String permission : requiredPermissions()) {
            if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private void requestRequiredPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(requiredPermissions(), REQUEST_PERMISSIONS);
        }
    }

    private String[] requiredPermissions() {
        List<String> permissions = new ArrayList<>();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions.add(Manifest.permission.BLUETOOTH_SCAN);
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT);
        }
        permissions.add(Manifest.permission.ACCESS_COARSE_LOCATION);
        permissions.add(Manifest.permission.ACCESS_FINE_LOCATION);
        return permissions.toArray(new String[0]);
    }

    private void updatePermissionUi() {
        permissionButton.setVisibility(hasRequiredPermissions() ? View.GONE : View.VISIBLE);
    }

    private void loadSavedBeacons() {
        SharedPreferences preferences = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String raw = preferences.getString(PREFS_BEACONS, "[]");
        savedBeacons.clear();
        try {
            JSONArray array = new JSONArray(raw);
            for (int index = 0; index < array.length(); index++) {
                JSONObject item = array.getJSONObject(index);
                String mac = normalizeMac(item.optString("mac"));
                String name = item.optString("name").trim();
                int color = item.optInt("color", palette[4]);
                if (mac != null && !name.isEmpty()) {
                    savedBeacons.add(new SavedBeacon(mac, name, color));
                }
            }
        } catch (JSONException ignored) {
            savedBeacons.clear();
        }
    }

    private void saveBeacons() {
        JSONArray array = new JSONArray();
        for (SavedBeacon beacon : savedBeacons) {
            JSONObject item = new JSONObject();
            try {
                item.put("mac", beacon.mac);
                item.put("name", beacon.name);
                item.put("color", beacon.color);
                array.put(item);
            } catch (JSONException ignored) {
                // JSONObject only receives primitive values here.
            }
        }
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
                .edit()
                .putString(PREFS_BEACONS, array.toString())
                .apply();
    }

    private SavedBeacon findSaved(String mac) {
        for (SavedBeacon beacon : savedBeacons) {
            if (beacon.mac.equals(mac)) {
                return beacon;
            }
        }
        return null;
    }

    private static String normalizeMac(String value) {
        if (value == null) {
            return null;
        }
        String compact = value.trim()
                .toUpperCase(Locale.US)
                .replace("-", "")
                .replace(":", "")
                .replace(" ", "");
        if (!compact.matches("[0-9A-F]{12}")) {
            return null;
        }
        StringBuilder builder = new StringBuilder(17);
        for (int index = 0; index < compact.length(); index += 2) {
            if (builder.length() > 0) {
                builder.append(':');
            }
            builder.append(compact, index, index + 2);
        }
        return builder.toString();
    }

    private TextView label(String text, int sp, int color, int style) {
        TextView textView = new TextView(this);
        textView.setText(text);
        textView.setTextSize(sp);
        textView.setTextColor(color);
        textView.setTypeface(Typeface.DEFAULT, style);
        return textView;
    }

    private LinearLayout.LayoutParams matchWrapParams(int bottomMargin, int topMargin) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        params.setMargins(0, topMargin, 0, bottomMargin);
        return params;
    }

    private GradientDrawable rounded(int color, int radius, int strokeColor, int strokeWidth) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(radius);
        if (strokeWidth > 0) {
            drawable.setStroke(strokeWidth, strokeColor);
        }
        return drawable;
    }

    private GradientDrawable swatchBackground(int color, boolean selected) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setShape(GradientDrawable.OVAL);
        drawable.setColor(color);
        drawable.setStroke(selected ? dp(4) : dp(1), selected ? Color.rgb(20, 24, 28) : Color.rgb(221, 223, 216));
        return drawable;
    }

    private int readableTextColor(int backgroundColor) {
        double luminance = (0.299 * Color.red(backgroundColor))
                + (0.587 * Color.green(backgroundColor))
                + (0.114 * Color.blue(backgroundColor));
        return luminance > 150 ? Color.rgb(22, 26, 30) : Color.WHITE;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static class SavedBeacon {
        final String mac;
        String name;
        int color;

        SavedBeacon(String mac, String name, int color) {
            this.mac = mac;
            this.name = name;
            this.color = color;
        }
    }

    private static class LiveBeacon {
        final String mac;
        final int rssi;
        final long lastSeenMs;

        LiveBeacon(String mac, int rssi, long lastSeenMs) {
            this.mac = mac;
            this.rssi = rssi;
            this.lastSeenMs = lastSeenMs;
        }
    }
}

package com.example.imageclassification;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.gson.Gson;

import java.util.HashMap;

public class ImageStorage {
    public static class ImageHashMap extends HashMap<String, String> {}
    private static final String KEY_PREFIX = "image_classification.image_storage.";
    private static final String IMAGES_FILE = KEY_PREFIX + "images_file";
    private static final String IMAGES_TABLE = KEY_PREFIX + "images_table";
    private static final String PRELOAD_FLAG = KEY_PREFIX + "preload_flag";
    public static SharedPreferences mPreferences;
    public static SharedPreferences.Editor mPreferencesEditor;
    private static Gson gson;

    public static void initialize(Context context) {
        mPreferences = context.getSharedPreferences(IMAGES_FILE, Context.MODE_PRIVATE);
        mPreferencesEditor = mPreferences.edit();
        gson = new Gson();
    }

    public static ImageHashMap getImagesMap() {
        return gson.fromJson(mPreferences.getString(IMAGES_TABLE, gson.toJson(new ImageHashMap())), ImageHashMap.class);
    }

    public static void saveImageMap(ImageHashMap map) {
        mPreferencesEditor.putString(IMAGES_TABLE, gson.toJson(map));
        mPreferencesEditor.apply();
    }
    public static void putImage(String name, Bitmap bitmap) {
        ImageHashMap map = getImagesMap();
        map.put(name, ImageManager.BitmapToBase64(bitmap));
        saveImageMap(map);
    }

    public static void setPreloadFlag() {
        mPreferencesEditor.putBoolean(PRELOAD_FLAG, true);
    }

    public static boolean isPreloaded() {
        return mPreferences.getBoolean(PRELOAD_FLAG, false);
    }
}

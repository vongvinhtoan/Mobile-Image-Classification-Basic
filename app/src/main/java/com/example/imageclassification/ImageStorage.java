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

    private static SharedPreferences getImagePreferences(Context context) {
        return context.getSharedPreferences(IMAGES_FILE, Context.MODE_PRIVATE);
    }

    public static ImageHashMap getImagesMap(Context context) {
        SharedPreferences mPreferences = getImagePreferences(context);
        Gson gson = new Gson();
        return gson.fromJson(mPreferences.getString(IMAGES_TABLE, gson.toJson(new ImageHashMap())), ImageHashMap.class);
    }

    public static void saveImageMap(Context context, ImageHashMap map) {
        Log.d("saveImageMap", String.valueOf(map.get("Toan").length()));
        SharedPreferences mPreferences = getImagePreferences(context);
        SharedPreferences.Editor mPreferencesEditor = mPreferences.edit();
        Gson gson = new Gson();
        mPreferencesEditor.putString(IMAGES_TABLE, gson.toJson(map));
        mPreferencesEditor.apply();
    }
    public static void putImage(Context context, String name, Bitmap bitmap) {
        ImageHashMap map = getImagesMap(context);
        map.put(name, ImageManager.BitmapToBase64(bitmap));
        saveImageMap(context, map);
    }
}

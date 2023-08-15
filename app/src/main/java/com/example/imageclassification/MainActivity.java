package com.example.imageclassification;

import static android.content.ContentValues.TAG;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDelegate;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.Manifest;

import com.example.imageclassification.ml.Facenet;
import com.example.imageclassification.ml.Facenet512;
import com.example.imageclassification.ml.Model;
import com.example.imageclassification.ml.Model1;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {
    public static final int REQUEST_CAMERA_FEATURE = 1;
    public static final int REQUEST_CLASSIFICATION = 2;
    public static final int REQUEST_UPDATE_DATA = 3;

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    EditText nameInput;
    int imageSize = 160;

    ImageStorage.ImageHashMap imageMap;
    HashMap<String, float[]> featureVectorMap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        Button updBtn = findViewById(R.id.add_data_btn);
        nameInput = findViewById(R.id.name_input);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, REQUEST_CLASSIFICATION);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_FEATURE);
                }
            }
        });

        updBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, REQUEST_UPDATE_DATA);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_FEATURE);
                }
            }
        });

        ImageStorage.initialize(this);
        initImageData();
    }

    private void initImageData() {
        imageMap = ImageStorage.getImagesMap();
        if (!ImageStorage.isPreloaded()) {
            imageMap.put("Nghiem", ImageManager.BitmapToBase64(BitmapFactory.decodeResource(getResources(), R.drawable.nghiem_image)));
            imageMap.put("Phuc", ImageManager.BitmapToBase64(BitmapFactory.decodeResource(getResources(), R.drawable.phuc_image)));
            imageMap.put("Toan", ImageManager.BitmapToBase64(BitmapFactory.decodeResource(getResources(), R.drawable.toan_image)));
            imageMap.put("Mai", ImageManager.BitmapToBase64(BitmapFactory.decodeResource(getResources(), R.drawable.mai_image)));
            ImageStorage.setPreloadFlag();
        }
        Log.d("initImageData", "Started");
        featureVectorMap = new HashMap<>();
        for (String name : imageMap.keySet()) {
            featureVectorMap.put(name, initFeatureVector(ImageManager.Base64ToBitmap(Objects.requireNonNull(imageMap.get(name)))));
        }
        Log.d("initImageData", "Done");
    }

    private float[] initFeatureVector(Bitmap image) {
        int dimension = Math.min(image.getWidth(), image.getHeight());
        image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
        image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

        try {
            float[] res;
            Facenet512 model = Facenet512.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 160, 160, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 160 * 160 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Facenet512.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            res = outputFeature0.getFloatArray();

            // Releases model resources if no longer used.
            model.close();

            return res;
        } catch (IOException e) {
            // TODO Handle the exception
        }
        return null;
    }
    private float[] initFeatureVector(int imageId) {
        return initFeatureVector(BitmapFactory.decodeResource(getResources(), imageId));
    }

    private float L2Dist(float[] a, float[] b) {
        float res = 0;
        float a_nor = 0;
        for(int i =0; i <a.length;i++) a_nor += a[i]*a[i];
        a_nor = (float)Math.pow(a_nor, 0.5);
        float b_nor = 0;
        for(int i =0; i <b.length;i++) b_nor += b[i]*b[i];
        b_nor = (float)Math.pow(b_nor, 0.5);
        for (int i=0; i<a.length; i++) res += Math.pow(a[i]/a_nor - b[i]/b_nor, 2);
        res = (float) Math.pow(res, 0.5);
        return res;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            assert data != null;
            Bitmap image = (Bitmap) Objects.requireNonNull(data.getExtras()).get("data");
            assert image != null;
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            if(requestCode == REQUEST_CLASSIFICATION) {
                classifyImage3(image);
            } else if (requestCode == REQUEST_UPDATE_DATA) {
                updateImageData(nameInput.getText().toString(), image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void updateImageData(String name, Bitmap bitmap) {
        featureVectorMap.put(name, initFeatureVector(bitmap));
        imageMap.put(name, ImageManager.BitmapToBase64(bitmap));
    }

    private HashMap<String, Float> generateResult(float[] feature) {
        HashMap<String, Float> res = new HashMap<>();
        for (String name : featureVectorMap.keySet()) {
            res.put(name, L2Dist(Objects.requireNonNull(featureVectorMap.get(name)), feature));
        }
        return res;
    }

    private void classifyImage3(Bitmap image) {
        try {
            Facenet512 model = Facenet512.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 160, 160, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Facenet512.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] feature = outputFeature0.getFloatArray();

            for(int i=0; i<feature.length; i++) {
                Log.i(TAG, "classifyImage2: " + String.valueOf(i) + ": " + String.valueOf(feature[i]));
            }

            HashMap<String, Float> genRes = generateResult(feature);

            String resName = null;
            for (String name : genRes.keySet()) {
                if (resName == null || genRes.get(resName) > genRes.get(name)) {
                    resName = name;
                }
            }

            result.setText(resName);

            String s = "";
            for (String name : genRes.keySet()) {
                s += String.format("%s: %.5f\n", name, genRes.get(name));
            }
            confidence.setText(s);

            Log.i(TAG, "classifyImage2: Shortest distant: " + resName);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
    private void classifyImage2(Bitmap image) {
        try {
            Facenet model = Facenet.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 160, 160, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Facenet.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] feature = outputFeature0.getFloatArray();


            for(int i=0; i<feature.length; i++) {
                Log.i(TAG, "classifyImage2: " + String.valueOf(i) + ": " + String.valueOf(feature[i]));
            }

            HashMap<String, Float> genRes = generateResult(feature);

            String resName = null;
            for (String name : genRes.keySet()) {
                if(resName == null || genRes.get(resName) > genRes.get(name)) {
                    resName = name;
                }
            }

            result.setText(resName);

            String s = "";
            for (String name : genRes.keySet()) {
                s += String.format("%s: %.5f\n", name, genRes.get(name));
            }
            confidence.setText(s);

            Log.i(TAG, "classifyImage2: Shortest distant: " + resName);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    private void classifyImage1(Bitmap bitmap) {
        try {
            Model1 model = Model1.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorImage image = TensorImage.fromBitmap(bitmap);

            // Runs model inference and gets result.
            Model1.Outputs outputs = model.process(image);
            List<Category> probability = outputs.getProbabilityAsCategoryList();

            int maxPos = 0;
            float maxScore = 0;
            for(int i=0; i<probability.size(); i++) {
                if(probability.get(i).getScore() > maxScore) {
                    maxScore = probability.get(i).getScore();
                    maxPos = i;
                }
            }

            String[] classes = {"Nghiem", "Phuc"};

            result.setText(classes[maxPos]);

            String s = "";
            for(int i = 0; i < classes.length; i++) {
                s += String.format("%s: %.1f%%\n", classes[i], probability.get(i).getScore() * 100);
            }
            confidence.setText(s);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    private void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            // Creates inputs for reference.
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Banana", "Orange", "Pen", "Sticky Notes"};
            result.setText(classes[maxPos]);

            String s = "";
            for(int i = 0; i < classes.length; i++){
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
            }
            confidence.setText(s);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onStop() {
        ImageStorage.saveImageMap(imageMap);
        super.onStop();
    }
}
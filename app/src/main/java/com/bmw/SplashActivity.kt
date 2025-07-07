package com.bmw

import android.animation.ObjectAnimator
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.bmw.databinding.ActivitySplashBinding

class SplashActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySplashBinding
    private val splashTimeOut: Long = 3000 // 3 seconds

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Enable edge-to-edge for modern look
        enableEdgeToEdge()
        
        binding = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Start animations
        startAnimations()
        
        // Navigate to MainActivity after splash timeout
        Handler(Looper.getMainLooper()).postDelayed({
            startMainActivity()
        }, splashTimeOut)
    }

    private fun enableEdgeToEdge() {
        // Hide system bars for immersive experience
        val windowInsetsController = ViewCompat.getWindowInsetsController(window.decorView)
        windowInsetsController?.let {
            it.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            it.hide(WindowInsetsCompat.Type.systemBars())
        }
    }

    private fun startAnimations() {
        // BMW Logo fade in and scale animation
        binding.ivBmwLogo.apply {
            alpha = 0f
            scaleX = 0.5f
            scaleY = 0.5f
            
            animate()
                .alpha(1f)
                .scaleX(1f)
                .scaleY(1f)
                .setDuration(1000)
                .setInterpolator(AccelerateDecelerateInterpolator())
                .start()
        }
        
        // App title slide up animation
        binding.tvAppTitle.apply {
            alpha = 0f
            translationY = 100f
            
            animate()
                .alpha(1f)
                .translationY(0f)
                .setDuration(800)
                .setStartDelay(500)
                .setInterpolator(AccelerateDecelerateInterpolator())
                .start()
        }
        
        // Subtitle fade in
        binding.tvSubtitle.apply {
            alpha = 0f
            
            animate()
                .alpha(1f)
                .setDuration(600)
                .setStartDelay(800)
                .start()
        }
        
        // Loading progress animation
        binding.progressBar.apply {
            alpha = 0f
            
            animate()
                .alpha(1f)
                .setDuration(400)
                .setStartDelay(1200)
                .start()
        }
        
        // Loading text animation
        binding.tvLoading.apply {
            alpha = 0f
            
            animate()
                .alpha(1f)
                .setDuration(400)
                .setStartDelay(1400)
                .start()
        }
        
        // AI status animation
        binding.tvAiStatus.apply {
            alpha = 0f
            
            animate()
                .alpha(1f)
                .setDuration(400)
                .setStartDelay(1600)
                .start()
        }
        
        // Pulsing animation for logo
        Handler(Looper.getMainLooper()).postDelayed({
            startLogoPulseAnimation()
        }, 1500)
    }
    
    private fun startLogoPulseAnimation() {
        val pulse = ObjectAnimator.ofFloat(binding.ivBmwLogo, "alpha", 1f, 0.7f, 1f)
        pulse.duration = 1500
        pulse.repeatCount = ObjectAnimator.INFINITE
        pulse.interpolator = AccelerateDecelerateInterpolator()
        pulse.start()
    }

    private fun startMainActivity() {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        
        // Add smooth transition
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        
        finish()
    }


}
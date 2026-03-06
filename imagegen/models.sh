#!/bin/bash

# ImageGen Model Management Script for Nvidia DGX Spark
# This script will manage AI models for image and video generation
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[MODELS]${NC} $1"
}

# Check if models directory exists
if [ ! -d "models" ]; then
    print_error "Models directory not found. Please run ./setup.sh first."
    exit 1
fi

# Load HF_TOKEN from .env file if it exists and HF_TOKEN is not already set
if [ -z "$HF_TOKEN" ] && [ -f ".env" ]; then
    export $(grep -v '^#' .env | grep HF_TOKEN | xargs)
fi

# Function to download with progress and verification
download_model() {
    local url="$1"
    local output_path="$2"
    local description="$3"
    local requires_auth="${4:-false}"
    
    print_status "Downloading $description..."
    
    if [ -f "$output_path" ]; then
        print_warning "File already exists: $output_path"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Skipping $description"
            return 0
        fi
    fi
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")"
    
    # Check for Hugging Face token if authentication is required
    if [ "$requires_auth" = "true" ]; then
        if [ -z "$HF_TOKEN" ] && [ ! -f "$HOME/.huggingface/token" ]; then
            print_error "Authentication required for $description"
            echo ""
            print_warning "Hugging Face token not found."
            echo "Please follow these steps:"
            echo "  1. Visit https://huggingface.co/settings/tokens"
            echo "  2. Create a new token with 'repo' read access"
            echo "  3. Copy the token"
            echo ""
            read -p "Paste your HF token here: " hf_token_input
            
            if [ -z "$hf_token_input" ]; then
                print_error "No token provided. Cannot continue."
                return 1
            fi
            
            # Save token to .env file
            if [ ! -f ".env" ]; then
                cat > .env << EOF
# Hugging Face API Token (required for gated models like Flux)
HF_TOKEN=$hf_token_input
EOF
                print_status "Token saved to .env file"
                chmod 600 .env  # Restrict permissions for security
            else
                # Check if HF_TOKEN already exists in .env
                if grep -q "^HF_TOKEN=" .env; then
                    sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$hf_token_input|" .env
                else
                    echo "HF_TOKEN=$hf_token_input" >> .env
                fi
                print_status "Token updated in .env file"
            fi
            
            export HF_TOKEN="$hf_token_input"
        fi
    fi
    
    # Download with wget, showing progress
    if command -v wget &> /dev/null; then
        if [ "$requires_auth" = "true" ] && [ -n "$HF_TOKEN" ]; then
            wget --progress=bar:force:noscroll --header="Authorization: Bearer $HF_TOKEN" -O "$output_path" "$url" 2>&1
        else
            wget --progress=bar:force:noscroll -O "$output_path" "$url" 2>&1
        fi
    elif command -v curl &> /dev/null; then
        if [ "$requires_auth" = "true" ] && [ -n "$HF_TOKEN" ]; then
            curl -L --progress-bar -H "Authorization: Bearer $HF_TOKEN" -o "$output_path" "$url" 2>&1
        else
            curl -L --progress-bar -o "$output_path" "$url" 2>&1
        fi
    else
        print_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        print_status "Downloaded $description successfully!"
    else
        print_error "Failed to download $description"
        rm -f "$output_path"
        return 1
    fi
}

# Function to list available models
list_models() {
    print_header "Installed models in each category:"
    echo ""
    
    echo "📁 Checkpoints:"
    find models/checkpoints/ -name "*.safetensors" -o -name "*.ckpt" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo ""
    
    echo "🎭 VAE:"
    find models/vae/ -name "*.safetensors" -o -name "*.ckpt" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo ""
    
    echo "🎯 LoRAs:"
    find models/loras/ -name "*.safetensors" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo ""
    
    echo "🎮 ControlNet:"
    find models/controlnet/ -name "*.safetensors" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo ""
    
    echo "📈 Upscale Models:"
    find models/upscale_models/ -name "*.pth" -o -name "*.safetensors" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo ""
    
    echo "🔤 Text Encoders (CLIP):"
    find models/clip/ -name "*.safetensors" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo ""
    
    echo "🧠 UNet Models:"
    find models/unet/ -name "*.safetensors" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo ""
}

# Function to install base models (essential for getting started)
install_base() {
    print_header "Installing base models for ComfyUI..."
    
    # Stable Diffusion v1.5 (most compatible)
    download_model \
        "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors" \
        "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors" \
        "Stable Diffusion v1.5 (FP16)"
    
    # VAE for better image quality
    download_model \
        "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors" \
        "models/vae/vae-ft-mse-840000-ema-pruned.safetensors" \
        "SD VAE (MSE)"
    
    # Basic upscaler
    download_model \
        "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth" \
        "models/upscale_models/4x_ESRGAN.pth" \
        "4x ESRGAN Upscaler"
    
    # Required ComfyUI models (z_image_turbo)
    print_status "Installing required ComfyUI models..."
    
    # Z Image Turbo Diffusion Model
    download_model \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
        "models/diffusion_models/z_image_turbo_bf16.safetensors" \
        "Z Image Turbo Diffusion Model"
    
    # Z Image Turbo Text Encoder
    download_model \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
        "models/text_encoders/qwen_3_4b.safetensors" \
        "Qwen Text Encoder"
    
    # Z Image Turbo VAE
    download_model \
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" \
        "models/vae/ae.safetensors" \
        "Z Image Turbo VAE"
    
    print_status "Base models installation complete!"
}

# Function to install SDXL models
install_sdxl() {
    print_header "Installing Stable Diffusion XL models..."
    
    # SDXL Base
    download_model \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
        "models/checkpoints/sd_xl_base_1.0.safetensors" \
        "SDXL Base 1.0"
    
    # SDXL Refiner
    download_model \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors" \
        "models/checkpoints/sd_xl_refiner_1.0.safetensors" \
        "SDXL Refiner 1.0"
    
    # SDXL VAE
    download_model \
        "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors" \
        "models/vae/sdxl_vae.safetensors" \
        "SDXL VAE (FP16 Fix)"
    
    print_status "SDXL models installation complete!"
}

# Function to install Flux models
install_flux() {
    print_header "Installing Flux models..."
    print_warning "Flux models require Hugging Face authentication"
    print_warning "If you haven't already, please set up your HF token:"
    print_warning "  huggingface-cli login"
    echo ""
    
    # Flux Schnell (fast generation model)
    download_model \
        "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors" \
        "models/unet/flux1-schnell.safetensors" \
        "Flux Schnell (Fast Generation)" \
        "true"
    
    # Flux VAE
    download_model \
        "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors" \
        "models/vae/flux_vae.safetensors" \
        "Flux VAE" \
        "true"
    
    # Flux Text Encoder
    download_model \
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
        "models/clip/clip_l.safetensors" \
        "Flux CLIP-L Text Encoder" \
        "true"
    
    download_model \
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
        "models/clip/t5xxl_fp16.safetensors" \
        "Flux T5-XXL Text Encoder (FP16)" \
        "true"
    
    print_status "Flux models installation complete!"
    print_status "Note: Flux models are optimized for fast, high-quality generation"
    print_status "      Perfect for DGX Spark's 128GB unified memory!"
}

# Function to install ControlNet models
install_controlnet() {
    print_header "Installing ControlNet models..."
    
    # Canny edge detection
    download_model \
        "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth" \
        "models/controlnet/control_v11p_sd15_canny.pth" \
        "ControlNet Canny"
    
    # Depth estimation
    download_model \
        "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" \
        "models/controlnet/control_v11f1p_sd15_depth.pth" \
        "ControlNet Depth"
    
    # OpenPose
    download_model \
        "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
        "models/controlnet/control_v11p_sd15_openpose.pth" \
        "ControlNet OpenPose"
    
    print_status "ControlNet models installation complete!"
}

# Function to install video models
install_video() {
    print_header "Installing Video generation models..."
    
    # Stable Video Diffusion
    download_model \
        "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors" \
        "models/checkpoints/svd_xt.safetensors" \
        "Stable Video Diffusion XT"
    
    print_status "Video models installation complete!"
}

# Function to install upscale models
install_upscale() {
    print_header "Installing upscale models..."
    
    # Real-ESRGAN 4x
    download_model \
        "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth" \
        "models/upscale_models/4x_ESRGAN.pth" \
        "4x ESRGAN"
    
    # Real-ESRGAN 2x
    download_model \
        "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth" \
        "models/upscale_models/2x_ESRGAN.pth" \
        "2x ESRGAN"
    
    # LDSR for high quality upscaling
    download_model \
        "https://huggingface.co/CompVis/ldsr-generic/resolve/main/model.ckpt" \
        "models/upscale_models/LDSR.ckpt" \
        "LDSR Upscaler"
    
    print_status "Upscale models installation complete!"
}

# Function to remove models
remove_models() {
    print_header "Model removal tool"
    echo ""
    echo "Available model categories:"
    echo "1. Checkpoints"
    echo "2. VAE"
    echo "3. LoRAs"
    echo "4. ControlNet"
    echo "5. Upscale Models"
    echo "6. All models (complete cleanup)"
    echo ""
    
    read -p "Select category to clean (1-6): " category
    
    case $category in
        1)
            print_warning "Removing all checkpoint models..."
            rm -rf models/checkpoints/*
            ;;
        2)
            print_warning "Removing all VAE models..."
            rm -rf models/vae/*
            ;;
        3)
            print_warning "Removing all LoRA models..."
            rm -rf models/loras/*
            ;;
        4)
            print_warning "Removing all ControlNet models..."
            rm -rf models/controlnet/*
            ;;
        5)
            print_warning "Removing all upscale models..."
            rm -rf models/upscale_models/*
            ;;
        6)
            print_warning "Removing ALL models..."
            read -p "Are you sure? This cannot be undone (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf models/*/
                print_status "All models removed."
            else
                print_status "Cleanup cancelled."
            fi
            ;;
        *)
            print_error "Invalid selection."
            exit 1
            ;;
    esac
    
    print_status "Cleanup complete!"
}

# Function to show disk usage
show_usage() {
    print_header "Model storage usage:"
    echo ""
    du -h models/ 2>/dev/null | sort -h
    echo ""
    echo "Total:"
    du -sh models/ 2>/dev/null || echo "0B models/"
}

# Function to update models
update_models() {
    print_header "Updating existing models..."
    print_warning "This will re-download all existing models."
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Re-run installations for existing model types
        if [ -n "$(find models/checkpoints/ -name "*.safetensors" 2>/dev/null)" ]; then
            install_base
        fi
        print_status "Update complete!"
    else
        print_status "Update cancelled."
    fi
}

# Main script logic
case "${1:-help}" in
    "list")
        list_models
        ;;
    "install")
        case "$2" in
            "base"|"")
                install_base
                ;;
            "sdxl")
                install_sdxl
                ;;
            "flux")
                install_flux
                ;;
            "controlnet")
                install_controlnet
                ;;
            "video")
                install_video
                ;;
            "upscale")
                install_upscale
                ;;
            "all")
                install_base
                install_sdxl
                install_flux
                install_controlnet
                install_upscale
                print_status "All model sets installed!"
                ;;
            *)
                print_error "Unknown model set: $2"
                echo "Available sets: base, sdxl, flux, controlnet, video, upscale, all"
                exit 1
                ;;
        esac
        ;;
    "remove"|"cleanup")
        remove_models
        ;;
    "usage"|"du")
        show_usage
        ;;
    "update")
        update_models
        ;;
    "help"|*)
        print_header "ImageGen Model Management for DGX Spark"
        echo ""
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  list              List all installed models"
        echo "  install [set]     Install model sets:"
        echo "    base            Essential models (SD 1.5, VAE, basic upscaler)"
        echo "    sdxl            Stable Diffusion XL models"
        echo "    flux            Flux Schnell models (fast, high-quality generation)"
        echo "    controlnet      ControlNet models for guided generation"
        echo "    video           Video generation models"
        echo "    upscale         High-quality upscaling models"
        echo "    all             Install all model sets"
        echo "  remove            Interactive model removal tool"
        echo "  usage             Show disk usage of models"
        echo "  update            Update existing models to latest versions"
        echo "  help              Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 install base     # Install essential models to get started"
        echo "  $0 install all      # Install everything (requires significant storage)"
        echo "  $0 list             # See what models are installed"
        echo "  $0 usage            # Check how much storage models are using"
        echo ""
        echo "Note: Models are downloaded to the models/ directory and can be large (1-7GB each)."
        echo "The DGX Spark's 128GB unified memory can handle multiple large models simultaneously."
        ;;
esac


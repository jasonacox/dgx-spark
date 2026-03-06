#!/bin/bash

# Script to archive NanoChat model checkpoints
# Archives all training phase checkpoints into timestamped backup directory
#
# Author: Jason Cox
# Date: 2025-12-06
# https://github.com/jasonacox/dgx-spark

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== NanoChat Checkpoint Archiver ===${NC}"
echo ""

# Configuration
CACHE_DIR="$HOME/.cache/nanochat"
ARCHIVE_BASE_DIR="$HOME/.cache/nanochat/archives"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="$ARCHIVE_BASE_DIR/checkpoint_archive_$TIMESTAMP"

# Checkpoint directories to archive
CHECKPOINT_DIRS=(
    "base_checkpoints"
    "mid_checkpoints"
    "chatsft_checkpoints"
    "chatrl_checkpoints"
)

# Parse command line arguments
DRY_RUN=0
COMPRESS=0
DELETE_AFTER=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --compress)
            COMPRESS=1
            shift
            ;;
        --delete-after)
            DELETE_AFTER=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run        Show what would be archived without actually doing it"
            echo "  --compress       Compress the archive to .tar.gz after archiving"
            echo "  --delete-after   Delete original checkpoints after successful archive"
            echo "  --help           Show this help message"
            echo ""
            echo "Archives are stored in: $ARCHIVE_BASE_DIR"
            echo ""
            echo "Example:"
            echo "  $0                    # Archive checkpoints"
            echo "  $0 --dry-run          # Preview what would be archived"
            echo "  $0 --compress         # Archive and compress to tar.gz"
            echo "  $0 --delete-after     # Archive and delete originals (frees space)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
    echo -e "${RED}Error: NanoChat cache directory not found: $CACHE_DIR${NC}"
    exit 1
fi

# Function to get directory size
get_dir_size() {
    du -sh "$1" 2>/dev/null | cut -f1
}

# Function to count files in directory
count_files() {
    find "$1" -type f 2>/dev/null | wc -l
}

# Check what exists
echo "Scanning for checkpoints..."
echo ""

FOUND_CHECKPOINTS=0
TOTAL_SIZE=0

for dir in "${CHECKPOINT_DIRS[@]}"; do
    checkpoint_path="$CACHE_DIR/$dir"
    if [ -d "$checkpoint_path" ]; then
        file_count=$(count_files "$checkpoint_path")
        if [ "$file_count" -gt 0 ]; then
            size=$(get_dir_size "$checkpoint_path")
            echo -e "${GREEN}✓${NC} Found: $dir ($size, $file_count files)"
            FOUND_CHECKPOINTS=$((FOUND_CHECKPOINTS + 1))
        else
            echo -e "${YELLOW}○${NC} Empty: $dir"
        fi
    else
        echo -e "${YELLOW}○${NC} Not found: $dir"
    fi
done

echo ""

if [ $FOUND_CHECKPOINTS -eq 0 ]; then
    echo -e "${YELLOW}No checkpoints found to archive.${NC}"
    exit 0
fi

if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}[DRY RUN MODE]${NC}"
    echo ""
fi

# Create archive directory
echo -e "${BLUE}Archive destination:${NC} $ARCHIVE_DIR"
echo ""

if [ $DRY_RUN -eq 0 ]; then
    mkdir -p "$ARCHIVE_DIR"
fi

# Archive each checkpoint directory
for dir in "${CHECKPOINT_DIRS[@]}"; do
    checkpoint_path="$CACHE_DIR/$dir"
    
    if [ -d "$checkpoint_path" ]; then
        file_count=$(count_files "$checkpoint_path")
        if [ "$file_count" -gt 0 ]; then
            echo -e "${BLUE}Archiving${NC} $dir..."
            
            if [ $DRY_RUN -eq 0 ]; then
                cp -r "$checkpoint_path" "$ARCHIVE_DIR/"
                echo -e "${GREEN}✓${NC} Copied to archive"
            else
                echo -e "${YELLOW}[DRY RUN]${NC} Would copy $checkpoint_path to $ARCHIVE_DIR/"
            fi
        fi
    fi
done

# Create archive metadata
if [ $DRY_RUN -eq 0 ]; then
    cat > "$ARCHIVE_DIR/ARCHIVE_INFO.txt" << EOF
NanoChat Checkpoint Archive
===========================

Archive Date: $(date '+%Y-%m-%d %H:%M:%S')
Archive Directory: $ARCHIVE_DIR
Original Location: $CACHE_DIR

Archived Checkpoints:
EOF

    for dir in "${CHECKPOINT_DIRS[@]}"; do
        checkpoint_path="$CACHE_DIR/$dir"
        if [ -d "$checkpoint_path" ]; then
            file_count=$(count_files "$checkpoint_path")
            if [ "$file_count" -gt 0 ]; then
                size=$(get_dir_size "$checkpoint_path")
                echo "  - $dir: $size ($file_count files)" >> "$ARCHIVE_DIR/ARCHIVE_INFO.txt"
            fi
        fi
    done
    
    echo "" >> "$ARCHIVE_DIR/ARCHIVE_INFO.txt"
    echo "To restore these checkpoints:" >> "$ARCHIVE_DIR/ARCHIVE_INFO.txt"
    echo "  cp -r $ARCHIVE_DIR/[checkpoint_dir] $CACHE_DIR/" >> "$ARCHIVE_DIR/ARCHIVE_INFO.txt"
fi

echo ""

# Compress archive if requested
if [ $COMPRESS -eq 1 ]; then
    TARBALL="$ARCHIVE_BASE_DIR/checkpoint_archive_$TIMESTAMP.tar.gz"
    
    echo -e "${BLUE}Compressing archive...${NC}"
    
    if [ $DRY_RUN -eq 0 ]; then
        tar -czf "$TARBALL" -C "$ARCHIVE_BASE_DIR" "checkpoint_archive_$TIMESTAMP"
        compressed_size=$(get_dir_size "$TARBALL")
        echo -e "${GREEN}✓${NC} Created: $TARBALL ($compressed_size)"
        
        # Remove uncompressed archive directory
        rm -rf "$ARCHIVE_DIR"
        echo -e "${GREEN}✓${NC} Removed uncompressed archive"
    else
        echo -e "${YELLOW}[DRY RUN]${NC} Would create: $TARBALL"
        echo -e "${YELLOW}[DRY RUN]${NC} Would remove uncompressed archive"
    fi
    
    FINAL_ARCHIVE="$TARBALL"
else
    FINAL_ARCHIVE="$ARCHIVE_DIR"
fi

echo ""

# Delete originals if requested
if [ $DELETE_AFTER -eq 1 ]; then
    echo -e "${YELLOW}Warning: About to delete original checkpoints!${NC}"
    echo "This will free up disk space but remove the originals from the cache."
    echo ""
    
    if [ $DRY_RUN -eq 0 ]; then
        echo -n "Are you sure you want to proceed? (yes/NO): "
        read -r response
        
        if [ "$response" = "yes" ]; then
            for dir in "${CHECKPOINT_DIRS[@]}"; do
                checkpoint_path="$CACHE_DIR/$dir"
                if [ -d "$checkpoint_path" ]; then
                    file_count=$(count_files "$checkpoint_path")
                    if [ "$file_count" -gt 0 ]; then
                        echo -e "${YELLOW}Deleting${NC} $dir..."
                        rm -rf "$checkpoint_path"
                        echo -e "${GREEN}✓${NC} Deleted"
                    fi
                fi
            done
            echo ""
            echo -e "${GREEN}✓${NC} Original checkpoints deleted"
        else
            echo -e "${BLUE}Skipped deletion${NC}"
        fi
    else
        echo -e "${YELLOW}[DRY RUN]${NC} Would delete original checkpoints after confirmation"
    fi
    
    echo ""
fi

# Summary
echo -e "${BLUE}=== Archive Complete ===${NC}"
echo ""

if [ $DRY_RUN -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Checkpoints archived to:"
    echo "  $FINAL_ARCHIVE"
    echo ""
    
    if [ $COMPRESS -eq 1 ]; then
        archive_size=$(get_dir_size "$FINAL_ARCHIVE")
        echo "Archive size: $archive_size"
    else
        archive_size=$(get_dir_size "$FINAL_ARCHIVE")
        echo "Archive size: $archive_size"
        echo ""
        echo "To compress later:"
        echo "  tar -czf checkpoint_archive_$TIMESTAMP.tar.gz -C $ARCHIVE_BASE_DIR checkpoint_archive_$TIMESTAMP"
    fi
    
    echo ""
    echo "To restore checkpoints:"
    if [ $COMPRESS -eq 1 ]; then
        echo "  tar -xzf $TARBALL -C $CACHE_DIR --strip-components=1"
    else
        echo "  cp -r $ARCHIVE_DIR/* $CACHE_DIR/"
    fi
    
    echo ""
    echo "To list all archives:"
    echo "  ls -lh $ARCHIVE_BASE_DIR"
else
    echo -e "${YELLOW}[DRY RUN]${NC} No changes made"
    echo ""
    echo "Run without --dry-run to perform the archive."
fi

echo ""

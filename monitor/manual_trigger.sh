# Get the FIP_TRAIN environment variable
FIP_TRAIN_VALUE=$(echo $FIP_TRAIN)

# Check if the environment variable exists
if [ -z "$FIP_TRAIN_VALUE" ]; then
    echo "Error: FIP_TRAIN environment variable is not set"
    exit 1
fi

# Print the value (optional)
echo "Retrieved FIP_TRAIN value: $FIP_TRAIN_VALUE"

# Construct the URL
URL="http://${FIP_TRAIN_VALUE}:8000/trigger"  # Adjust the port and endpoint as needed

# Make the curl request
echo "Making request to: $URL"
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"type": trigger, "index": index}' \
     $URL
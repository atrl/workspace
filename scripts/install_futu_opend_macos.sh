#!/bin/bash
set -euo pipefail

# Install Futu OpenD for macOS automatically.
# The .zip / .dmg URLs return 404, but the .tar.gz URL works.

VERSION="${VERSION:-10.2.6208}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/futu_opend}"
URL="https://softwaredownload.futunn.com/Futu_OpenD_${VERSION}_Mac.tar.gz"
ARCHIVE="/tmp/Futu_OpenD_${VERSION}_Mac.tar.gz"

echo "=> Downloading Futu OpenD ${VERSION} for macOS..."
echo "   URL: ${URL}"
curl -L -o "${ARCHIVE}" "${URL}"

echo "=> Extracting to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
tar -xzf "${ARCHIVE}" -C "${INSTALL_DIR}"

echo "=> Removing quarantine attributes..."
OPEND_APP="${INSTALL_DIR}/Futu_OpenD_${VERSION}_Mac/Futu_OpenD_${VERSION}_Mac/FutuOpenD.app"
if [ -d "${OPEND_APP}" ]; then
    xattr -r -d com.apple.quarantine "${OPEND_APP}" 2>/dev/null || true
fi

echo "=> Done. OpenD installed at:"
echo "   ${OPEND_APP}"
echo ""
echo "Next steps:"
echo "1. Edit your credentials in:"
echo "   ${INSTALL_DIR}/Futu_OpenD_${VERSION}_Mac/Futu_OpenD_${VERSION}_Mac/FutuOpenD.xml"
echo "2. Start OpenD:"
echo "   ${OPEND_APP}/Contents/MacOS/FutuOpenD"
echo "3. Verify connectivity:"
echo "   futu_venv/bin/python production/check_futu_opend.py"

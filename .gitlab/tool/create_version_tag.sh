#!/bin/bash
if git ls-remote --tags origin "refs/tags/v${VERSION}" | grep -q "refs/tags/v${VERSION}"; then
    echo "Tag v${VERSION} already exists, skipping..."
else
    echo "Creating new tag v${VERSION}..."
    git config user.email "gitlab-ci@espressif.com"
    git config user.name "GitLab CI"
    git tag -a "v${VERSION}" -m "Release version ${VERSION}"
    git push origin "v${VERSION}"
fi

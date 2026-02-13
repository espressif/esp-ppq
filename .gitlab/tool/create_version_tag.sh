#!/bin/bash
if git ls-remote --tags origin "refs/tags/${VERSION}" | grep -q "refs/tags/${VERSION}"; then
    echo "Tag ${VERSION} already exists, skipping..."
else
    echo "Creating new tag ${VERSION}..."
    git config user.email "gitlab-ci@espressif.com"
    git config user.name "GitLab CI"
    git tag -a "${VERSION}" -m "Release version ${VERSION}"
    git push origin "${VERSION}"
fi

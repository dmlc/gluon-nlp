Release Checklist
=================

Below is the checklist for releasing a new minor version of GluonNLP:

- Creat a new release branch $major.$minor.x with commits from the master branch
- Bump the version in master branch to $major.$minor+1.$patch.dev
- Bump the version in release branch to $major.$minor.$patch
- Draft the release note, highlight important talks/models/features, as well as breaking changes
- Publish the release on Github
- Refresh website
- Prepare pip package
- Make annoucement (Twitter, etc)

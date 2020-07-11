import coins


def trainModels(translate=False, multiclass=False, split='mean', dropPercent=5):
    # LOAD RAW DATA
    ipip = coins.io.getPreprocessedRawData('ipip')
    images = coins.io.getPreprocessedRawData('images')
    imageLabels = coins.io.getPreprocessedRawData('imageLabels')

    # BUILD DEFINED DATAFRAMES
    dfPersonality = coins.dfcreation.createPersonality(ipip)
    dfImageDescriptions = coins.dfcreation.createImageDescriptions(images)
    dfImageRatings = coins.dfcreation.createImageRatings(images, train=True)
    dfSocioDemographics = coins.dfcreation.createSocioDemographics([ipip, images])
    dfImageContents = coins.dfcreation.createImageContents(images, imageLabels)
    
    if(translate == True):
        # OPTION 1: Analyze and save imageDescriptions (needs credentials, costs money)
        dfImageDescriptions = coins.nluTranslation.translateToEnglish(dfImageDescriptions)
        dfImageDescriptions = coins.nluTranslation.analyzeEnglishSentimentAndEmotions(dfImageDescriptions)
        dfImageDescriptions = coins.nluTranslation.fillImageDescriptions(dfImageDescriptions)
        coins.io.saveAnalyzedImageDescriptions(dfImageDescriptions)
    else:
        # OPTION 2: Load existing analyzed imageDescriptions
        dfImageDescriptions = coins.io.loadAnalyzedImageDescriptions()
    
    # PREPARE DATAFRAME VALUES
    dfPersonality = coins.correlation.preparePersonality(dfPersonality, multiclass=multiclass, split=split, train=True)
    dfImageDescriptions = coins.correlation.prepareImageDescriptions(dfImageDescriptions, multiclass=multiclass, split=split, train=True)
    dfSocioDemographics, dropList = coins.correlation.prepareSocioDemographics(dfSocioDemographics, dropPercent)

    resultsPersonality = coins.classification.findBestClassifier([dfImageRatings, dfSocioDemographics, dfImageContents, dfImageDescriptions], dfPersonality, "dfPersonality", inputFeatureCombination=False, printProgress=False)
    resultsImageRatings = coins.classification.findBestClassifier([dfPersonality, dfSocioDemographics, dfImageContents, dfImageDescriptions], dfImageRatings, "dfImageRatings", inputFeatureCombination=False, printProgress=False)
    resultsSocioDemographics = coins.classification.findBestClassifier([dfPersonality, dfImageRatings, dfImageContents, dfImageDescriptions], dfSocioDemographics, "dfSocioDemographics", inputFeatureCombination=False, printProgress=False)
    resultsImageContents = coins.classification.findBestClassifier([dfPersonality, dfImageRatings, dfSocioDemographics, dfImageDescriptions], dfImageContents, "dfImageContents", inputFeatureCombination=False, printProgress=False)
    resultsImageDescriptions = coins.classification.findBestClassifier([dfPersonality, dfImageRatings, dfSocioDemographics, dfImageContents], dfImageDescriptions, "dfImageDescriptions", inputFeatureCombination=False, printProgress=False)

    return "Alle Model wurden erfolgreich erstellt. Du findest sie im Ordner 'output/modelResults'."

def calculateCorrelations(translate=False, multiclass=False, split='mean', dropPercent=5):
    # LOAD RAW DATA
    ipip = coins.io.getPreprocessedRawData('ipip')
    images = coins.io.getPreprocessedRawData('images')
    imageLabels = coins.io.getPreprocessedRawData('imageLabels')

    # BUILD DEFINED DATAFRAMES
    dfPersonality = coins.dfcreation.createPersonality(ipip)
    dfImageDescriptions = coins.dfcreation.createImageDescriptions(images)
    dfImageRatings = coins.dfcreation.createImageRatings(images, train=True)
    dfSocioDemographics = coins.dfcreation.createSocioDemographics([ipip, images])
    dfImageContents = coins.dfcreation.createImageContents(images, imageLabels)
    
    if(translate == True):
        # OPTION 1: Analyze and save imageDescriptions (needs credentials, costs money)
        dfImageDescriptions = coins.nluTranslation.translateToEnglish(dfImageDescriptions)
        dfImageDescriptions = coins.nluTranslation.analyzeEnglishSentimentAndEmotions(dfImageDescriptions)
        dfImageDescriptions = coins.nluTranslation.fillImageDescriptions(dfImageDescriptions)
        coins.io.saveAnalyzedImageDescriptions(dfImageDescriptions)
    else:
        # OPTION 2: Load existing analyzed imageDescriptions
        dfImageDescriptions = coins.io.loadAnalyzedImageDescriptions()
    
    # PREPARE DATAFRAME VALUES
    dfPersonality = coins.correlation.preparePersonality(dfPersonality, multiclass=multiclass, split=split, train=True)
    dfImageDescriptions = coins.correlation.prepareImageDescriptions(dfImageDescriptions, multiclass=multiclass, split=split, train=True)
    dfSocioDemographics, dropList = coins.correlation.prepareSocioDemographics(dfSocioDemographics, dropPercent)

    # CALCULATE CORRELATIONS AND P-VALUES
    _, _, _, pPersonalitySocioDemographics, cPersonalitySocioDemographics = coins.correlation.calculateCorrWithPValue(dfPersonality, dfSocioDemographics)
    _, _, _, pPersonalityImageDescriptions, cPersonalityImageDescriptions = coins.correlation.calculateCorrWithPValue(dfPersonality, dfImageDescriptions)
    _, _, _, pPersonalityImageRatings, cPersonalityImageRatings = coins.correlation.calculateCorrWithPValue(dfPersonality, dfImageRatings)
    _, _, _, pSocioDemographicsImageDescriptions, cSocioDemographicsImageDescriptions = coins.correlation.calculateCorrWithPValue(dfSocioDemographics, dfImageDescriptions)
    _, _, _, pSocioDemographicsImageRatings, cSocioDemographicsImageRatings = coins.correlation.calculateCorrWithPValue(dfSocioDemographics, dfImageRatings)
    _, _, _, pImageDescriptionsImageRatings, cImageDescriptionsImageRatings = coins.correlation.calculateCorrWithPValue(dfImageDescriptions, dfImageRatings)
    _, _, _, pImageContentsSocioDemographics, cImageContentsSocioDemographics = coins.correlation.calculateCorrWithPValue(dfImageContents, dfSocioDemographics)
    _, _, _, pImageContentsImageDescriptions, cImageContentsImageDescriptions = coins.correlation.calculateCorrWithPValue(dfImageContents, dfImageDescriptions)
    _, _, _, pImageContentsImageRatings, cImageContentsImageRatings = coins.correlation.calculateCorrWithPValue(dfImageContents, dfImageRatings)
    _, _, _, pImageContentsPersonality, cImageContentsPersonality = coins.correlation.calculateCorrWithPValue(dfImageContents, dfPersonality)

    # EXTRACT SIGNIFICANT CORRELATIONS
    sPersonalitySocioDemographics = coins.correlation.extractSignificantCorrelations(pPersonalitySocioDemographics, cPersonalitySocioDemographics)
    sPersonalityImageDescriptions = coins.correlation.extractSignificantCorrelations(pPersonalityImageDescriptions, cPersonalityImageDescriptions)
    sPersonalityImageRatings = coins.correlation.extractSignificantCorrelations(pPersonalityImageRatings, cPersonalityImageRatings)
    sSocioDemographicsImageDescriptions = coins.correlation.extractSignificantCorrelations(pSocioDemographicsImageDescriptions, cSocioDemographicsImageDescriptions)
    sSocioDemographicsImageRatings = coins.correlation.extractSignificantCorrelations(pSocioDemographicsImageRatings, cSocioDemographicsImageRatings)
    sImageDescriptionsImageRatings = coins.correlation.extractSignificantCorrelations(pImageDescriptionsImageRatings, cImageDescriptionsImageRatings)
    sImageContentsSocioDemographics = coins.correlation.extractSignificantCorrelations(pImageContentsSocioDemographics, cImageContentsSocioDemographics)
    sImageContentsImageDescriptions = coins.correlation.extractSignificantCorrelations(pImageContentsImageDescriptions, cImageContentsImageDescriptions)
    sImageContentsImageRatings = coins.correlation.extractSignificantCorrelations(pImageContentsImageRatings, cImageContentsImageRatings)
    sImageContentsPersonality = coins.correlation.extractSignificantCorrelations(pImageContentsPersonality, cImageContentsPersonality)

    # SAVE SIGNIFICANT CORRELATIONS
    coins.io.saveSignificantCorrelations(sPersonalitySocioDemographics, 'personality_socioDemographics')
    coins.io.saveSignificantCorrelations(sPersonalityImageDescriptions, 'personality_imageDescriptions')
    coins.io.saveSignificantCorrelations(sPersonalityImageRatings, 'personality_imageRatings')
    coins.io.saveSignificantCorrelations(sSocioDemographicsImageDescriptions, 'socioDemographics_imageDescriptions')
    coins.io.saveSignificantCorrelations(sSocioDemographicsImageRatings, 'imageRatings_socioDemographics')
    coins.io.saveSignificantCorrelations(sImageDescriptionsImageRatings, 'imageDescriptions_imageRatings')
    coins.io.saveSignificantCorrelations(sImageContentsSocioDemographics, 'imageContents_socioDemographics')
    coins.io.saveSignificantCorrelations(sImageContentsImageDescriptions, 'imageContents_imageDescriptions')
    coins.io.saveSignificantCorrelations(sImageContentsImageRatings, 'imageContents_imageRatings')
    coins.io.saveSignificantCorrelations(sImageContentsPersonality, 'imageContents_personality')

    return "Alle Correlationen wurden erfolgreich berechnet. Du findest sie im Ordner 'output/correlations'."

def predictNewData(targetDataFrame):
    results = coins.classification.prediction.predictNewData(targetDataFrame)
    return results
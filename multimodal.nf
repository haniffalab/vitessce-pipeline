#!/usr/bin/env/ nextflow

import groovy.json.*

include { ome_zarr_metadata } from "./main.nf"
include { warnParams } from "./main.nf"

nextflow.enable.dsl=2

params.outdir = ""
params.copy_raw = true
params.description = ""

version="0.5.3"
verbose_log=true
outdir_with_version = "${params.outdir.replaceFirst(/\/*$/, "")}\/${version}"

config_map = [
    title: params.title ?: "",
    description: params.description ?: "",
    url: params.url ?: "http://localhost/",
    extend_feature_name: params.extend_feature_name ?: null
]

def copyFile (inputFile, outdir) {
    def outfile = file(outdir) / "$inputFile.name"
    if (outfile.exists()){
        log.warn "File $inputFile.name already exists in $outdir. Skipping copy."
    }
    else {
        inputFile.copyTo(outdir)
    }
    return inputFile
}

def parseExtendFeature (extend_feature){
    if (!extend_feature){
        return [ file("NO_FT"), [:] ]
    }
    else if (extend_feature instanceof String ){
        return [
            file(extend_feature.endsWith(".h5ad") ? extend_feature : "NO_FT"),
            [:]
        ]
    }
    else if (extend_feature instanceof Map){
        if (extend_feature["path"]){
            if (!(extend_feature.path instanceof String && extend_feature.path.endsWith(".h5ad"))){
                error "Invalid value for `extend_feature.path`. Expecting an .h5ad file."
            }
            return [ file(extend_feature.path), extend_feature.args ?: [:] ]
        }
        else {
            error "Invalid map for `extend_feature`. Expecting key `path`."
        }
    }
    error "Invalid value for `extend_feature`"
}

process process_label {
    tag "${label_image}"
    debug verbose_log

    publishDir outdir_with_version, mode:"copy"

    input:
    tuple val(dataset), path(label_image), val(offset)

    output:
    tuple val(dataset), path("${reindexed_label_image}")

    script:
    basename = label_image.baseName
    reindexed_label_image = "reindexed-${basename}.ome.zarr"
    """
    integrate_image.py \
        --label_image_path ${label_image} \
        --offset ${offset} \
        --out_filename ${reindexed_label_image}
    """
}


process process_anndata {
    tag "${anndata}"
    debug verbose_log

    publishDir outdir_with_version, mode:"copy"

    input:
    tuple val(dataset), path(anndata), val(offset), val(features), path(features_file), val(features_args)

    output:
    tuple val(dataset), path("*")

    script:
    features_str = features
        ? "--features ${features_file.name != 'NO_FT' ? features_file : features}"
        : ""
    feature_name_str = config_map.extend_feature_name
        ? "--concat_feature_name ${config_map.extend_feature_name}"
        : ""
    args_str = features_file.name != 'NO_FT' && features_args
        ? "--args '" + new JsonBuilder(features_args).toString() + "'" : ""
    """
    integrate_anndata.py reindex_and_concat \
        --path ${anndata} \
        --offset ${offset} \
        ${features_str} ${args_str}
    """
}


process intersect_anndatas {
    debug verbose_log

    publishDir outdir_with_version, mode:"copy"

    input:
    path(anndatas)

    output:
    path("*")

    script:
    """
    integrate_anndata.py intersect_features ${anndatas}
    """
}

process Build_multimodal_config {
    tag "${project}"
    label 'build_config'
    debug verbose_log
    cache false

    publishDir outdir_with_version, mode: "copy"

    input:
    tuple val(project), val(config_map), val(datasets)

    output:
    path("${project}-multimodal-config.json")

    script:
    url_str = config_map.url?.trim() ? "--url \"${config_map.url.trim()}\"" : ""
    extended_features_str = config_map.extend_feature_name
        ? "--extended_features ${config_map.extend_feature_name}"
        : ""
    datasets_str = new JsonBuilder(datasets).toString()
    """
    build_config_multimodal.py \
        --project "${project}" \
        --datasets '${datasets_str}' \
        ${extended_features_str} \
        ${url_str} \
        --title "${config_map.title}" \
        --description "${config_map.description}"
    """
}


workflow {

    warnParams()

    file(outdir_with_version).mkdirs()

    Channel.from(params.data)
        .multiMap{ it ->
            info: [it.dataset, it.obs_type ?: "cell", it.is_spatial ?: false, it.vitessce_options ?: [:]]
            raws : [it.dataset, it.raw_image] // not processed but necessary for writing config
            labels : [it.dataset, it.label_image, it.offset]
            adatas : [it.dataset, file(it.anndata), it.offset, it.extend_feature,
                *parseExtendFeature(it.extend_feature)
            ]
        }
        .set{data}

    // Filter null raw image
    data.raws.filter{ it[1] }
        .map{ [it[0], params.copy_raw ? copyFile(file(it[1]), outdir_with_version) : file(it[1])] }
        .set{raw_images}
        
    // Process labels filtered out nulls
    process_label(data.labels.filter { it[1] }.map{ [it[0], file(it[1]), it[2]] })

    // Reindex and concatenate anndata matrices
    // filename will be used to join output from intersection
    process_anndata(data.adatas)
    process_anndata.out
        .map{ [it[1].baseName.toString(), it[0], it[1] ] }
        .set{reid_output}

    // Perform intersection on list of anndatas
    // filename without 'intersect-' is used to match the dataset it comes from
    // as we cannot get paired dataset-file items as output
    // because output is a list of files (they're processed together)
    intersect_anndatas(process_anndata.out.collect{ it[1] })
    intersect_anndatas.out
        .flatMap()
        .map{
            // strip the extra 'intersect-' from file basename
            def basename = it.baseName.split("intersect-")[-1]
            [basename, it]
        }
        .set{intersect_output}
    
    // Get ome zarr metadata from raw and label images
    ome_zarr_metadata(
        raw_images.map{[
            it[0],
            "raw",
            file("${it[1].toString().replaceFirst(/\/*$/, '')}/OME/METADATA.ome.xml")
        ]}.concat(
            process_label.output.map{[
                it[0],
                "label",
                file("${it[1].toString().replaceFirst(/\/*$/, '')}/OME/METADATA.ome.xml")
            ]}
        )
    ).set{images_md}

    // Structure image data to join to corresponding dataset data
    raw_images.map{[it[0], "raw", it[1]]}
        .concat(process_label.output.map{[it[0], "label", it[1]]})
        .join(images_md, by: [0,1])
        .map { dataset, type, path, md ->
            [
                dataset, type, [path: path.name, md: new JsonSlurper().parseText(md.trim())]
            ]
        }
        .groupTuple(by: [0,1])
        .map { dataset, type, img ->
            [dataset, [type: type, img: img]]
        }
        .branch { dataset, data ->
            raw: data.type == "raw"
            label: data.type == "label"
        }
        .set{imgs}
    
    // Image data structure to be able to perform join:
    // [dataset, raw:[raw_imgs], label:[label_imgs]]
    imgs.raw
        .join(imgs.label, remainder: true)
        .map { dataset, raw_data, label_data -> [
            dataset,
            [
                raw: raw_data ? raw_data.img : [],
                label: label_data ? label_data.img : []

            ]
        ]}
        .set{img_map}
    
    // Join reindexed and intersected outputs
    // to get [dataset, intersected_file]
    reid_output
        .join(intersect_output, by: 0)
        .map{ [it[1], it[3].name] }
        .groupTuple(by: 0)
        .set{files}
    
    // tmp_data to replace potential nulls for empty list and map
    data.info
        .join(files, by:0)
        .join(img_map, by: 0, remainder: true)
        .map{[*it[0..3], it[4] ?: [], it[5] ?: [:]]}
        .set{tmp_data}

    // make map of each dataset
    tmp_data.map{[
            "${it[0]}":
            [
		obs_type: it[1], is_spatial: it[2], options: it[3],
                file_paths: it[4], images: it[5]
            ]
        ]}
        .reduce{ a,b -> a + b } // merge maps
        .map{[params.project, config_map, it]}
        .set{data_for_config}
    
    Build_multimodal_config(data_for_config)

}

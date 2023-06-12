import os
import uuid

from dotenv import load_dotenv
from flask import Flask, flash, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from segmentation.predict import predict_segmentation

app = Flask(__name__)

STATIC_DIR = "./static/"
TMP_DIR = "tmp/"
FILE_STORAGE_DIR = STATIC_DIR + TMP_DIR

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
PARAMS_NOT_FOUND_RESPONSE = app.response_class(
    response='{"response": "204 Parameters are not found"}',
    status=204,
    mimetype="application/json",
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model = request.form["model"]
        file_to_segment = request.files["file_to_segment"]

        if not model or not file_to_segment:
            flash("All fields are required")
        else:
            img_id = str(uuid.uuid4())
            inp_path = os.path.join(
                FILE_STORAGE_DIR,
                img_id + "_" + secure_filename(file_to_segment.filename),
            )
            file_to_segment.save(inp_path)

            tt_output_filename = "tt_{}.png".format(img_id)
            tf_output_filename = "tf_{}.png".format(img_id)
            ft_output_filename = "ft_{}.png".format(img_id)
            ff_output_filename = "ff_{}.png".format(img_id)
            predict_segmentation(
                checkpoints_path=model_paths[model],
                inp=inp_path,
                out_fnames=[
                    FILE_STORAGE_DIR + tt_output_filename,
                    FILE_STORAGE_DIR + tf_output_filename,
                    FILE_STORAGE_DIR + ft_output_filename,
                    FILE_STORAGE_DIR + ff_output_filename,
                ],
                all_visualisations=True,
                class_names=["Background", "Passport", "Photo", "MRZ"],
            )

            image_links = {
                "tt": tt_output_filename,
                "tf": tf_output_filename,
                "ft": ft_output_filename,
                "ff": ff_output_filename,
            }
            return render_template(
                "index.html",
                image_links=image_links,
                selected_model=model,
                model_names=model_names,
                tmp_dir=TMP_DIR
            )
    return render_template("index.html", model_names=model_names)


@app.route("/api/v1/download/<file_name>")
def download(file_name):
    return send_from_directory(
        FILE_STORAGE_DIR,
        file_name,
        mimetype="image/png",
        download_name=file_name,
        as_attachment=True,
    )


@app.route("/api/v1/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return PARAMS_NOT_FOUND_RESPONSE
    file = request.files["file"]
    if not file and not allowed_file(file.filename):
        return PARAMS_NOT_FOUND_RESPONSE

    file_id = str(uuid.uuid4())
    file_extension = get_extension(file.filename)
    filename = file_id + "." + file_extension
    file.save(os.path.join(FILE_STORAGE_DIR, filename))
    return {"file_name": filename}


@app.route("/api/v1/models", methods=["GET"])
def models():
    return model_names


@app.route("/api/v1/segment", methods=["GET"])
def segment():
    input_filename = request.args["file_name"]
    segmentation_model_key = request.args["model_key"]
    show_legends = request.args.get("show_legends", default=False, type=lambda v: v.lower() == 'true')
    overlay_image = request.args.get("overlay_image", default=False, type=lambda v: v.lower() == 'true')

    if not input_filename or not segmentation_model_key:
        return PARAMS_NOT_FOUND_RESPONSE

    output_filename = str(uuid.uuid4()) + ".png"
    predict_segmentation(
        checkpoints_path=model_paths[segmentation_model_key],
        inp=os.path.join(FILE_STORAGE_DIR, secure_filename(input_filename)),
        out_fname=os.path.join(FILE_STORAGE_DIR, output_filename),
        show_legends=show_legends,
        overlay_img=overlay_image,
        class_names=["Background", "Passport", "Photo", "MRZ"],
    )
    return send_from_directory(
        FILE_STORAGE_DIR,
        output_filename,
        mimetype="image/png",
        download_name=output_filename,
        as_attachment=True,
    )


def get_extension(filename):
    return filename.rsplit(".", 1)[1].lower()


def allowed_file(filename):
    return "." in filename and get_extension(filename) in ALLOWED_EXTENSIONS


model_paths = {}
model_names = {}
load_dotenv()
for key, val in os.environ.items():
    # DOC_SEGMENTATION.MODEL.UNET.PATH=../models/unet/unet/unet
    PREFIX = "DOC_SEGMENTATION.MODEL."
    if not key.startswith(PREFIX):
        continue
    model_key = key.split(".")[2]
    if key.endswith("PATH"):
        model_paths[model_key] = val
    elif key.endswith("NAME"):
        model_names[model_key] = val

os.chdir("./app")
app.run(host="0.0.0.0", debug=False)

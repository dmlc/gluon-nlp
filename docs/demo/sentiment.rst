Sentiment Analysis
==================

Introduction
------------

Sentiment Analysis predicts whether an input is positive or negative. The model is based on BERT base, and are trained on the binary classification setting of the Stanford Sentiment Treebank. It achieves about 93.4% accuracy on the dev set.

Demo
----

You can either input a sentence into the textbox or select one sample from the select control.

.. raw:: html

    <label for="select">Enter text or </label>
    <select id="select" name="octane" style="
            display: block;
            font-size: 14px;
            font-family: sans-serif;
            font-weight: 700;
            color: #444;
            line-height: 1.3;
            padding: .6em 1.4em .5em .8em;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            margin: 0;
            border: 2px solid rgba(23, 141, 201, 0.66);
            box-shadow: 1px 1px 1px 1px rgba(0,0,0,.05);
            border-radius: 1em;
            -moz-appearance: none;
            -webkit-appearance: none;
            appearance: none;
            background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23007CB2%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E'),
              linear-gradient(to bottom, #ffffff 0%,#ffffff 100%);
            background-repeat: no-repeat, repeat;
            background-position: right .7em top 50%, 0 0;
            background-size: .65em auto, 100%;">
      <option>Choose an example...</option>
      <option value="Happy New Year!">Happy New Year!</option>
      <option value="That person seems to be in anger now.">That person seems to be in anger now.</option>
    </select>

    <div class="mdl-textfield mdl-js-textfield" style="width: 90%">
      <input class="mdl-textfield__input" type="text"
             id="input" name="data" value="Input a sentence..."
             style="border-bottom:1px solid rgba(23, 141, 201, 0.9);">
    </div>
    <input type="button" value="Try" class="mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect"
           style="color: rgba(23, 141, 201, 1);">
    <div id="result">
      Result will appear here.
    </div>

    <script type="text/javascript">
    $("#select").change(function () {
        let sentence = $(this).val();
        $("#input").val(sentence)
    });

    $("select+div+input").click(function () {
        let request_url = 'http://52.12.156.84:8888/bert_sst/predict?data=["' +
            $("#input").val() + '"]';
        $.ajax({
            url: request_url, success: function (result) {
                if (result[4] === '0')
                    $("#result").html("Negative Sentiment");
                else $("#result").html("Positive Sentiment");
            }
        });
    });
    </script>
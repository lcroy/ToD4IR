var socket = io.connect('http://' + document.domain + ':' + location.port);

var domain = 'delivery';
var domain_set = 'null';
var user_utterance = '';

// list of parameters related to delivery
var del_sa_area = 'null';
var del_sa_location = 'null';
var del_sa_object = 'null';

// list of parameters related to assembly
var asb_sa_producttype = 'null';
var asb_sa_quantity = 'null';


$(document).ready(function() {
    $( '#max_side_domains' ).show();
    $( '#s_message' ).show();
    $( '#user_side_task_specification' ).hide();
});

function checkmode(element){
  if (element.checked){
     $( '#max_side_domains' ).hide();
     $( '#s_message' ).hide();
     $( '#user_side_task_specification' ).show();
  } else{
    $( '#max_side_domains' ).show();
    $( '#s_message' ).show();
    $( '#user_side_task_specification' ).hide();
  }
}

// listening on tab changing event.
$(function () {
             $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
                 var target = $(e.target).attr("href");
                 var temp_domain = target.replace("#max_","");
                 if (temp_domain === 'delivery'){
                     domain_set = { "assembly": false, "delivery": true, "position": false, "relocation": false };
                     domain = 'delivery';
                 } else if (temp_domain === 'assembly'){
                     domain_set = { "assembly": true, "delivery": false, "position": false, "relocation": false };
                     domain = 'assembly';
                 } else if (temp_domain === 'position'){
                     domain_set = { "assembly": false, "delivery": false, "position": true, "relocation": false };
                     domain = 'position';
                 } else if (temp_domain === 'relocation'){
                     domain_set = { "assembly": false, "delivery": false, "position": false, "relocation": true };
                     domain = 'relocation';
                 }
                 // clear all the slots on the page
                // clear delivery
                $('#area').val('');
                $('#location').val('');
                $('#sender').val('');
                $('#recipient').val('');
                $('#object').val('');
                $('#color').val('');
                $('#size').val('');
                $("#message").val();
                $("#s_message").val();
                $("#area_res").text('');
                $("#location_res").text('');
                del_sa_area = 'null';
                del_sa_location = 'null';
                del_sa_object = 'null';

                // clear assembly
                $('#producttype').val('');
                $('#quantity').val('');
                $("#prod_color").val();
                $("#prod_size").val();
                $("#prod_style").text('');
                asb_sa_producttype = 'null';
                asb_sa_quantity = 'null';
             });
         })


// obtain time for message
function gettzdate(){
    var current_time = new Date().toLocaleTimeString();
    // if ($('#usermode').is(':checked')){
    //   alert("this is a test")
    // }
    return current_time ;
}

//search the database
function get_area() {
            $.ajax({
              url:'/get_area/',
              data: { "area": $('#area').val().toLowerCase()},
              type:"GET",
              dataType:'json',
                success:function (data) {
                    $.each(data,function(k,v) {
                        if (v == "detected"){
                            document.getElementById("area_res").textContent = "Great, " + $('#area').val().toLowerCase() + " is found.";
                            del_sa_area = 'detected';
                        }else if (v == "undetected"){
                            document.getElementById("area_res").textContent = "Sorry, " + $('#area').val().toLowerCase() + " is not found.";
                            del_sa_area = 'undetected'
                        }else{
                            document.getElementById("area_res").textContent = "Please enter the area first."
                            del_sa_area = 'null';
                        }
                    })
                }
            })
        };

//search the database
function get_location() {
            $.ajax({
              url:'/get_location/',
              data: { "area": $('#area').val().toLowerCase(),"location": $('#location').val().toLowerCase()},
              type:"GET",
              dataType:'json',
                success:function (data) {
                    $.each(data,function(k,v) {
                        if (v == "detected"){
                          document.getElementById("location_res").textContent = $('#area').val().toLowerCase() + " and " + $('#location').val().toLowerCase() + " are found.";
                          del_sa_location = 'detected';
                        }else if (v == "undetected"){
                            document.getElementById("location_res").textContent = "Area and Loation are not related.";
                            del_sa_location = 'undetected';
                        }else if (v =="Need Area and Location"){
                            document.getElementById("location_res").textContent = " Area and Location can not be empty.";
                            del_sa_area = 'null';
                            del_sa_location = 'null';
                        }else if (v =="Need Area"){
                            document.getElementById("location_res").textContent = "Need area for search location."
                        }else if (v =="Need Location"){
                            document.getElementById("location_res").textContent = "Location can not be empty.";
                            del_sa_location = 'null';
                        }
                    })
                }
            })
        };

//search the database
function get_product() {
            $.ajax({
              url:'/get_prod/',
              data: { "prod": $('#producttype').val()},
              type:"GET",
              dataType:'json',
                success:function (data) {
                    $.each(data,function(k,v) {
                        if (v == "detected"){
                            document.getElementById("producttype_res").textContent = "Great, " + $('#producttype').val() + " is found.";
                            asb_sa_producttype = 'detected';
                        }else if (v == "undetected"){
                            document.getElementById("producttype_res").textContent = "Sorry, " + $('#producttype').val() + " is not found.";
                            asb_sa_producttype = 'undetected'
                        }else{
                            document.getElementById("producttype_res").textContent = "Please enter the product type first."
                            asb_sa_producttype = 'null';
                        }
                    })
                }
            })
        };

//end conversation
function get_end_conv() {
            $.ajax({
              url:'/get_end_conv/',
              data: { "end": 'yes'},
              type:"GET",
              dataType:'json',
                success:function (data) {
                    $.each(data,function(k,v) {
                        if (v == "done"){
                            // clear page
                            if (domain == 'delivery'){
                                $('#area').val('');
                                $('#location').val('');
                                $('#sender').val('');
                                $('#recipient').val('');
                                $('#object').val('');
                                $('#color').val('');
                                $('#size').val('');
                                $("#message").val('');
                                $("#s_message").val('');
                                $("#area_res").text('');
                                $("#location_res").text('');
                                del_sa_area = 'null';
                                del_sa_location = 'null';
                                del_sa_object = 'null';
                            } else if (domain == 'assembly'){
                                $('#producttype').val('');
                                $('#quantity').val('');
                                $("#prod_color").val('');
                                $("#prod_size").val('');
                                $("#prod_style").val('');
                                asb_sa_producttype = 'null';
                                asb_sa_quantity = 'null';
                            }
                        }
                    })
                }
            })
        };


socket.on( 'connect', function() {
    socket.emit( 'my event', {
        data: 'User Connected is',
        flag: 'connection'
    } )
    var form = $( 'form' ).on( 'submit', function( e ) {
        e.preventDefault()
        let t_res = $("#message").val();
        user_utterance = $("#message").val();
        let s_res = $("#s_message").val();
        let speaker = '';
        let slots = '';

        if ($('#usermode').is(':checked')){
            speaker = "user";
            slots = {'user': user_utterance};
        } else {
            speaker = "max";
            // set up turn info.
            if (domain == 'delivery') {
                slots = {
                    "user": user_utterance,
                    "system": t_res,
                    "s_system": s_res,
                    "slots": {
                        "assembly": {
                            "DB_request": {
                                "req": {
                                    "producttype": ""
                                },
                                "opt": {}
                            },
                            "T_inform": {
                                "req": {
                                    "product": "",
                                    "quantity": ""
                                },
                                "opt": {
                                    "color": "",
                                    "style": "",
                                    "size": ""
                                },
                                "type": ""
                            }
                        },
                        "delivery": {
                            "DB_request": {
                                "req": {
                                    "area": $('#area').val() == ''?"not_mentioned":$('#area').val(),
                                    "location": $('#location').val() == ''?"not_mentioned":$('#location').val()
                                },
                                "opt": {
                                    "sender": $('#sender').val() == ''?"not_mentioned":$('#sender').val(),
                                    "recipient": $('#recipient').val() == ''?"not_mentioned":$('#recipient').val()
                                }
                            },
                            "T_inform": {
                                "req": {
                                    "object": $('#object').val() == ''?"not_mentioned":$('#object').val()
                                },
                                "opt": {
                                    "color": $('#color').val() == ''?"not_mentioned":$('#color').val(),
                                    "size": $('#size').val() == ''?"not_mentioned":$('#size').val()
                                },
                                "type": "delivery"
                            }
                        },
                        "position": {
                            "DB_request": {
                                "req": {
                                    "position_name": ""
                                },
                                "opt": {}
                            },
                            "T_inform": {
                                "req": {
                                    "operation": ""
                                },
                                "opt": {},
                                "type": ""
                            }
                        },
                        "relocation": {
                            "DB_request": {
                                "req": {
                                    "object": ""
                                },
                                "opt": {}
                            },
                            "T_inform": {
                                "req": {},
                                "opt": {
                                    "color": "",
                                    "size": "",
                                    "from": "",
                                    "to": ""
                                },
                                "type": ""
                            }
                        }
                    },
                    "search_result": {
                        "area": del_sa_area,
                        "location": del_sa_location,
                        "object": $('#object').val() == ''?"null":"detected"
                    }
                }
            };
            if (domain == 'assembly') {
                slots = {
                    "user": user_utterance,
                    "system": t_res,
                    "s_system": s_res,
                    "slots": {
                        "assembly": {
                            "DB_request": {
                                "req": {
                                    "producttype": $('#producttype').val() == ''?"not_mentioned":$('#producttype').val()
                                },
                                "opt": {}
                            },
                            "T_inform": {
                                "req": {
                                    "quantity": $('#quantity').val() == ''?"not_mentioned":$('#quantity').val()
                                },
                                "opt": {
                                    "color": $('#prod_color').val() == ''?"not_mentioned":$('#prod_color').val(),
                                    "style": $('#prod_size').val() == ''?"not_mentioned":$('#prod_size').val(),
                                    "size": $('#prod_style').val() == ''?"not_mentioned":$('#prod_style').val()
                                },
                                "type": "assembly"
                            }
                        },
                        "delivery": {
                            "DB_request": {
                                "req": {
                                    "area": "",
                                    "location": ""
                                },
                                "opt": {
                                    "sender": "",
                                    "recipient": ""
                                }
                            },
                            "T_inform": {
                                "req": {
                                    "object": ""
                                },
                                "opt": {
                                    "color": "",
                                    "size": ""
                                },
                                "type": ""
                            }
                        },
                        "position": {
                            "DB_request": {
                                "req": {
                                    "location": ""
                                },
                                "opt": {}
                            },
                            "T_inform": {
                                "req": {
                                    "operation": ""
                                },
                                "opt": {},
                                "type": ""
                            }
                        },
                        "relocation": {
                            "DB_request": {
                                "req": {
                                    "object": ""
                                },
                                "opt": {}
                            },
                            "T_inform": {
                                "req": {},
                                "opt": {
                                    "color": "",
                                    "size": "",
                                    "from": "",
                                    "to": ""
                                },
                                "type": ""
                            }
                        }
                    },
                    "search_result": {
                        "producttype": asb_sa_producttype,
                        "quantity": $('#quantity').val() == ''?"null":"detected"
                    }
                }
            }
        }
        socket.emit( 'my event', {
            domain : domain,
            domain_set: domain_set,
            slots : slots,
            t_res : t_res,
            s_res : s_res,
            speaker : speaker,
            flag: 'response'
        } )
        $("#message").val( '' ).focus();
        $("#s_message").val( '' )
    } )
} )


socket.on('my response', function( msg ) {
  if (msg.message !== undefined) {

    if (msg.message !==""){

      if ($('#usermode').is(':checked')) {

        if (msg.speaker == "max") {
          $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data\"><img src=\"/static/data/image/max.png\" alt=\"Max\"> " +
              "<span class=\"message-data-time\">" + gettzdate() + "</span></div><div class=\"message other-message\">"
              + msg.message + "</div></li>")
        } else if (msg.speaker == "user") {
          $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data text-right\"><span class=\"message-data-time\">" + gettzdate() +
              "</span><img src=\"/static/data/image/user.png\" alt=\"User\"></div><div class=\"message other-message float-right\">" + msg.message + "</div></li>")
        }
      } else {
        if (msg.speaker == "user") {
          $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data\"><img src=\"/static/data/image/user.png\" alt=\"User\"> " +
              "<span class=\"message-data-time\">" + gettzdate() + "</span></div><div class=\"message other-message\">"
              + msg.message + "</div></li>")
        } else if (msg.speaker == "max") {
          $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data text-right\"><span class=\"message-data-time\">" + gettzdate() +
              "</span><img src=\"/static/data/image/max.png\" alt=\"Max\"></div><div class=\"message other-message float-right\">" + msg.message + "</div></li>")
        }
      }
    } else {
      alert("Please enter the message before you post :)")
    }
  }
})




